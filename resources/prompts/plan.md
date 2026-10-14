Propose a sequential plan using only supplied versioned tool cards. Dependencies and
output references must point to earlier steps. Match each input schema exactly; keep
objects and arrays as JSON values. Use literal values or RFC 6901 JSON Pointer
bindings. Leave action_ref null because Python assigns business action identity.
Repair validation errors while preserving the requested goal. Never bypass an
unresolved write with a fresh action. Return only the Plan JSON contract.
