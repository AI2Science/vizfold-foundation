use sea_orm::DbErr;
use serde_json::Value;

pub fn require_json_object(field_name: &str, raw: &str) -> Result<Value, DbErr> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| DbErr::Custom(format!("{field_name} must be valid JSON: {error}")))?;

    if !value.is_object() {
        return Err(DbErr::Custom(format!("{field_name} must be a JSON object")));
    }

    Ok(value)
}

pub fn reject_unknown_keys(field_name: &str, schema: &Value, params: &Value) -> Result<(), DbErr> {
    let Some(params_object) = params.as_object() else {
        return Err(DbErr::Custom(format!("{field_name} must be a JSON object")));
    };

    let allowed_keys = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .keys()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
        });

    if let Some(allowed_keys) = allowed_keys
        && let Some(unexpected) = params_object
            .keys()
            .find(|key| !allowed_keys.contains(key.as_str()))
    {
        return Err(DbErr::Custom(format!(
            "{field_name} contains unsupported key '{unexpected}'"
        )));
    }

    Ok(())
}
