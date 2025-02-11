use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use house_price_predictor::aws::download_model_from_s3;
use house_price_predictor::model::{load_xgboost_model, Model};
use log::info;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use xgboost::DMatrix;
#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    bucket_name_s3: String,
    #[arg(short, long)]
    key_s3: String,
}

/// App state that will be shared across all workers of my actix server.

#[derive(Clone)]
struct AppState {
    model: Arc<Model>,
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

#[derive(Serialize)]
struct PredictResponse {
    prediction: f32,
}

/// Tranfrom a JSON Payload into a Dmatrix.
/// Returns an error if transformation get fails.

fn transform_features_payload_to_dmatrix(
    payload: &web::Json<PredictRequest>,
) -> anyhow::Result<DMatrix> {
    // transform the payload into a slice of floating like &[f32]
    let features: Vec<f32> = [
        payload.crim,
        payload.zn,
        payload.indus,
        payload.chas,
        payload.nox,
        payload.rm,
        payload.age,
        payload.dis,
        payload.rad,
        payload.tax,
        payload.ptratio,
        payload.b,
        payload.lstat,
    ]
    .iter()
    .map(|f| *f as f32)
    .collect();

    let dmatrix_features = DMatrix::from_dense(&features, 1)?;

    Ok(dmatrix_features)
}

#[post("/predict")]
async fn predict(payload: web::Json<PredictRequest>, data: web::Data<AppState>) -> impl Responder {
    info!("Predict health point called");
    info!("Features sent by the client:{:?}", payload);

    // let model_metdata = data.model.get_attribute_names().unwrap();
    // info!("Model metadata:{:?}", model_metdata);
    // Transform the payload into a DMatrix
    let dmatrix_features = transform_features_payload_to_dmatrix(&payload).unwrap();

    // Use the model and the `dmatrix_features` to generate a prediction
    let model = &data.model;
    let prediction = model.predict(&dmatrix_features).unwrap()[0];

    // build the response struct with the prediction
    let prediction_response = PredictResponse {
        prediction: prediction,
    };

    // Return the response as a JSON payload
    web::Json(prediction_response)
}

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logger

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Dowloading the model from s3 Bucket as a local file.
    let args = Args::parse();
    // Download the model from S3 into a local file
    let model_path = download_model_from_s3(&args.bucket_name_s3, &args.key_s3).await?;

    info!("Starting API...");

    HttpServer::new(move || {
        // Load the model into memory
        let model = load_xgboost_model(&model_path).unwrap();

        // Create the state data structure that will be shared across all workers
        let app_state = AppState {
            model: Arc::new(model),
        };

        App::new()
            .app_data(web::Data::new(app_state))
            .service(health)
            .service(predict)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;

    Ok(())
}
