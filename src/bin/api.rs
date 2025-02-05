use actix_web::{get, web, App, HttpResponse, HttpServer, Responder};
use log::{info, warn};

// Health check endpint
// Returns an 200 Ok response if the API is healthy.

#[get("/health")]
async fn health() -> impl Responder {
    info!("Health check endpoint called");
    HttpResponse::Ok().body("I am healthy!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    info!("Starting the API...");

    HttpServer::new(|| App::new().service(health))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await?;
    Ok(())
}
