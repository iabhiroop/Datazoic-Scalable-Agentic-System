"""Immutable payload references; persist essential provider status before optional offload."""

import asyncio
import base64
import hashlib
from uuid import UUID, uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException
from sqlalchemy import select

from app.db.models import PayloadReference
from app.db.repositories import CatalogRepository, RunRepository
from app.db.session import Database
from app.security import Principal
from app.settings import Settings


class StorageError(RuntimeError):
    """Sanitized storage boundary error; never propagate signed URLs or credentials."""


class PayloadStore:
    def __init__(self, database: Database, settings: Settings, *, client=None) -> None:
        self.database = database
        self.settings = settings
        self.bucket = settings.object_storage_bucket
        if client is None:
            if not (
                settings.object_storage_endpoint
                and settings.object_storage_access_key
                and settings.object_storage_secret_key
            ):
                raise ValueError("Object storage configuration is incomplete")
            client = boto3.client(
                "s3",
                endpoint_url=settings.object_storage_endpoint,
                aws_access_key_id=settings.object_storage_access_key.get_secret_value(),
                aws_secret_access_key=settings.object_storage_secret_key.get_secret_value(),
                region_name="us-east-1",
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=5,
                    read_timeout=15,
                    retries={"max_attempts": 1},
                ),
            )
        self.client = client

    async def call(self, method: str, **arguments):
        try:
            return await asyncio.to_thread(getattr(self.client, method), **arguments)
        except (BotoCoreError, ClientError):
            raise StorageError("Object storage request failed") from None

    async def setup(self) -> None:
        try:
            await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)
        except ClientError as error:
            if str(error.response.get("Error", {}).get("Code")) not in ("404", "NoSuchBucket"):
                raise StorageError("Object storage bucket check failed") from None
            await self.call("create_bucket", Bucket=self.bucket)
        except BotoCoreError:
            raise StorageError("Object storage bucket check failed") from None

    async def upload(
        self, data: bytes, content_type: str, run_id: UUID | None, principal: Principal
    ) -> PayloadReference:
        if not data or len(data) > self.settings.document_max_bytes:
            raise ValueError("Payload is empty or exceeds the configured size limit")
        required = "runs:write" if run_id else "knowledge:manage"
        if required not in principal.scopes:
            raise HTTPException(403, "Payload upload permission is required")
        if run_id:
            async with self.database.session() as session:
                await RunRepository(session).get(run_id, principal)
        identity = uuid4()
        key = f"{principal.tenant_id}/{run_id or 'documents'}/{identity}"
        checksum = hashlib.sha256(data).hexdigest()
        await self.call(
            "put_object",
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            IfNoneMatch="*",
            ContentMD5=base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode(),
            Metadata={"sha256": checksum},
        )
        stored = await self.call("head_object", Bucket=self.bucket, Key=key)
        if (
            stored["ContentLength"] != len(data)
            or stored.get("Metadata", {}).get("sha256") != checksum
        ):
            raise StorageError("Uploaded payload integrity check failed")
        reference = PayloadReference(
            payload_id=identity,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            run_id=run_id,
            object_key=key,
            content_type=content_type,
            size=len(data),
            checksum=checksum,
        )
        async with self.database.session() as session, session.begin():
            session.add(reference)
            await session.flush()
        return reference

    async def authorized_reference(
        self, payload_id: UUID, principal: Principal
    ) -> PayloadReference:
        async with self.database.session() as session:
            reference = await session.scalar(
                select(PayloadReference).where(
                    PayloadReference.tenant_id == principal.tenant_id,
                    PayloadReference.payload_id == payload_id,
                )
            )
            if reference is None:
                raise HTTPException(404, "Payload not found")
            if reference.run_id:
                if "runs:read" not in principal.scopes:
                    raise HTTPException(403, "Payload read permission is required")
                await RunRepository(session).get(reference.run_id, principal)
            elif reference.user_id == principal.user_id and "knowledge:manage" in principal.scopes:
                pass
            elif "knowledge:read" in principal.scopes:
                documents = await CatalogRepository(
                    session, allowed_adapters=set()
                ).active_documents(principal, [])
                if not any(doc.payload_ref == payload_id for doc in documents):
                    raise HTTPException(404, "Payload not found")
            else:
                raise HTTPException(404, "Payload not found")
            if not reference.object_key.startswith(f"{principal.tenant_id}/"):
                raise StorageError("Invalid scoped payload reference")
            return reference

    async def retrieve(self, payload_id: UUID, principal: Principal) -> bytes:
        reference = await self.authorized_reference(payload_id, principal)
        result = await self.call("get_object", Bucket=self.bucket, Key=reference.object_key)
        body = result["Body"]
        try:
            data = await asyncio.to_thread(body.read, reference.size + 1)
        finally:
            body.close()
        if len(data) != reference.size or hashlib.sha256(data).hexdigest() != reference.checksum:
            raise StorageError("Stored payload integrity check failed")
        return data

    async def signed_url(self, payload_id: UUID, principal: Principal) -> str:
        reference = await self.authorized_reference(payload_id, principal)
        return await self.call(
            "generate_presigned_url",
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": reference.object_key},
            ExpiresIn=300,
        )

    async def aclose(self) -> None:
        await asyncio.to_thread(self.client.close)

    async def cleanup_orphans(self) -> int:
        raise NotImplementedError(
            "Retention job; preserve pending financial and checkpoint evidence"
        )
