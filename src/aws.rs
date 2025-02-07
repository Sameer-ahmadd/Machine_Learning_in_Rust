use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::Client;

// Create an AWS S3 client. Loads the necessaray configuration from the env.

pub async fn create_s3_client() -> Client {
    let region_provider = RegionProviderChain::default_provider().or_else("us-west-2");
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    Client::new(&config)
}
/// Pushes model to S3 Bucket.
pub async fn pushes_model_to_s3(
    path_to_model: &str,
    bucket_name: &str,
    key: &str,
) -> anyhow::Result<()> {
    // Create an S3 client so i can talk to S3.

    let client = create_s3_client().await;

    // Load the model file into memory
    let model_file_bytes = std::fs::read(path_to_model)?;

    // Upload the model file to the S3 bucket
    // TODO: make this value a parameter to this function
    //  let bucket_name = "xgboost-rust";
    // let key = "boston_housing_model.bin";

    let _result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(model_file_bytes.into())
        .send()
        .await?;

    Ok(())
}

/// Download the model from S3 Bucket into memory as a Local file.
pub async fn download_model_from_s3(bucket_name: &str, key: &str) -> anyhow::Result<(String)> {
    let client = create_s3_client().await;

    //  let bucket_name = "xgboost-rust";
    // let key = "boston_housing_model.bin";

    // First we download the content of the model file from S3 into memory
    let download_path = "downloaded_model.bin";
    let resp = client
        .get_object()
        .bucket(bucket_name)
        .key(key)
        .send()
        .await?;
    let data = resp.body.collect().await?.into_bytes();

    // Save the downloaded bytes to a file on disk
    std::fs::write(download_path, data)?;

    Ok(download_path.to_string())
}
