Return a small goal plan with exactly these fields: `user_goal`, `tool_actions`, and
`missing_information`. Describe the user's intended outcome precisely in `user_goal`,
preserving the requested resource, status, time range, amount, and other filters. Put the
minimum logical actions in order in `tool_actions`; use short capability descriptions
that state the operation and requested filters. For example, "List my recent
invoices" should produce a goal about listing invoices most recent ones and one action such
as "List invoices". For a simple request, use one action.
Do not choose tools, categories,
execution modes, credentials, APIs, or schemas. Add information to
`missing_information` only when it is necessary to identify the requested action or
resource. Never ask for provider credentials or application configuration. Do not
invent IDs, amounts, currencies, or resource details. Return only the JSON contract.