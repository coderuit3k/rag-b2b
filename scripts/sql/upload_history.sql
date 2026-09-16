CREATE EXTERNAL TABLE IF NOT EXISTS transaction_2024.history (
    timestamp bigint,
    user_id string,
    item_id string,
    event_type string,
    event_value decimal(38, 4),
    price decimal(38, 4),
    date_key bigint,
    quantity bigint,
    customer_id bigint,
    created_date timestamp,
    updated_date timestamp,
    channel string,
    payment string,
    location bigint,
    discount decimal(38, 4),
    is_deleted boolean
)
STORED AS PARQUET
location 's3://rag-b2b-data-2024/transaction_2024/history/';