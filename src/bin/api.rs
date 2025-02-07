use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use house_price_predictor::aws::download_model_from_s3;
use log::info;
use serde::Deserialize;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    bucket_name_s3: String,
    #[arg(short, long)]
    key_s3: String,
}

// Health check endpint
// Returns an 200 Ok response if the API is healthy.

#[get("/health")]
async fn health() -> impl Responder {
    info!("Health check endpoint called");
    HttpResponse::Ok().body("I am healthy!")
}

#[derive(Deserialize, Debug)]
struct PredictRequest {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    b: f64,
    lstat: f64,
}

#[post("/predict")]
async fn predict(payload: web::Json<PredictRequest>) -> impl Responder {
    info!("Predict health point called");
    info!("Features sent by the client:{:?}", payload);

    HttpResponse::Ok().body("Prediction")
}

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logger

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Dowloading the model from s3 Bucket as a local file.
    let args = Args::parse();
    let model_path = download_model_from_s3(&args.bucket_name_s3, &args.key_s3).await?;

    info!("Starting the API...");

    HttpServer::new(|| App::new().service(health).service(predict))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await?;
    Ok(())
}
