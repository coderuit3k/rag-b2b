CREATE EXTERNAL TABLE IF NOT EXISTS transaction_2024.user (
    customer_id bigint,
    gender string,
    location bigint,
    province string,
    timestamp bigint,
    created_date timestamp,
    updated_date timestamp,
    is_deleted boolean,
    sync_status_id bigint,
    last_sync_date timestamp,
    sync_error_message string,
    region string,
    location_name string,
    install_app string,
    install_date bigint,
    district string,
    user_id string
)
STORED AS PARQUET
location 's3://rag-b2b-data-2024/transaction_2024/user/';