use axum::routing::get;
use axum::Json;
use axum::{extract::State, Router};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use serde_json::json;
use std::sync::mpsc;
use std::thread;
use tokio;

#[tokio::main]
async fn main() {
    let (request_tx, request_rx) = mpsc::channel::<(Vec<u32>, mpsc::Sender<u32>)>();

    thread::spawn(move || {
        const BUFFER_SIZE: usize = 1024;

        let device = CudaDevice::new(0).unwrap();
        let ptx = compile_ptx(include_str!("kernel.cu")).unwrap();
        device.load_ptx(ptx.clone(), "", &["sum_buffer"]).unwrap();
        let func = device.get_func("", "sum_buffer").unwrap();
        let cfg = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut buffer = device.alloc_zeros(BUFFER_SIZE).unwrap();

        while let Ok((message, response_tx)) = request_rx.recv() {
            if message.len() > BUFFER_SIZE {
                response_tx.send(0).unwrap();
                continue;
            }
            let mut host_buffer = [0u32; BUFFER_SIZE];
            host_buffer[..message.len()].copy_from_slice(&message);
            device
                .htod_copy_into(host_buffer.to_vec(), &mut buffer)
                .unwrap();
            let result = device.alloc_zeros(1).unwrap();

            unsafe {
                func.clone()
                    .launch(cfg, (&mut buffer, BUFFER_SIZE as u32, &result))
            }
            .unwrap();

            let result_host = device.dtoh_sync_copy(&result).unwrap();
            response_tx.send(result_host[0]).unwrap();
        }
    });

    let request_tx = std::sync::Arc::new(tokio::sync::Mutex::new(request_tx));

    let app = Router::new()
        .route("/process", get(process_handler))
        .with_state(request_tx);
    println!("Server running on http://localhost:4444");
    let listener = tokio::net::TcpListener::bind("localhost:4444")
        .await
        .unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn process_handler(
    State(request_tx): State<
        std::sync::Arc<tokio::sync::Mutex<mpsc::Sender<(Vec<u32>, mpsc::Sender<u32>)>>>,
    >,
) -> Json<serde_json::Value> {
    let (once_tx, once_rx) = mpsc::channel();
    let request_tx = request_tx.lock().await;
    request_tx.send((vec![1, 2, 3, 4, 5], once_tx)).unwrap();
    json!({ "result": once_rx.recv().unwrap() }).into()
}
