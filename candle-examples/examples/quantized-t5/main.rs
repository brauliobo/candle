#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use candle_transformers::models::quantized_t5 as t5;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, api::sync::ApiRepo, Repo, RepoType};
use tokenizers::Tokenizer;

use axum::{
    response::IntoResponse,
    http::StatusCode,
    Router, Json,
};
use serde_json::Value;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The model size to use.
    #[arg(long, default_value = "t5-small")]
    which: Which,
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let device = Device::new_cuda(0)?;
        let default_model = "lmz/candle-quantized-t5".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, "main".to_string()),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &args.config_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("config.json")?,
                Which::FlanT5Small => api.get("config-flan-t5-small.json")?,
                Which::FlanT5Base => api.get("config-flan-t5-base.json")?,
                Which::FlanT5Large => api.get("config-flan-t5-large.json")?,
                Which::FlanT5Xl => api.get("config-flan-t5-xl.json")?,
                Which::FlanT5Xxl => api.get("config-flan-t5-xxl.json")?,
            },
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &args.weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("model.gguf")?,
                Which::FlanT5Small => api.get("model-flan-t5-small.gguf")?,
                Which::FlanT5Base => api.get("model-flan-t5-base.gguf")?,
                Which::FlanT5Large => api.get("model-flan-t5-large.gguf")?,
                Which::FlanT5Xl => api.get("model-flan-t5-xl.gguf")?,
                Which::FlanT5Xxl => api.get("model-flan-t5-xxl.gguf")?,
            },
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let device = Device::new_cuda(0)?;
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename, &device)?;
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let builder = Arc::new(builder);
    let device = Arc::clone(&builder).device.clone();
    let model = builder.build_model()?;
    let model = Arc::new(Mutex::new(model));

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    fn generate_text(
        args: &Args,
        prompt: String,
        builder: Arc<T5ModelBuilder>,
        tokenizer: &mut Tokenizer,
        device: &Device,
        model: Arc<Mutex<t5::T5ForConditionalGeneration>>,
    ) -> Result<String, anyhow::Error> {
        let mut result = String::new();
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;

        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let mut model = model.lock().unwrap();
        let model = &mut *model;
        let mut output_token_ids = [builder
            .config
            .decoder_start_token_id
            .unwrap_or(builder.config.pad_token_id) as u32]
        .to_vec();
        let temperature = if args.temperature <= 0. {
            None
        } else {
            Some(args.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(299792458, temperature, args.top_p);
        let encoder_output = model.encode(&input_token_ids)?;
        let start = std::time::Instant::now();

        for index in 0.. {
            if output_token_ids.len() > 512 {
                break;
            }
            let decoder_token_ids = if index == 0 || !builder.config.use_cache {
                Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
            } else {
                let last_token = *output_token_ids.last().unwrap();
                Tensor::new(&[last_token], device)?.unsqueeze(0)?
            };
            let logits = model
                .decode(&decoder_token_ids, &encoder_output)?
                .squeeze(0)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &output_token_ids[start_at..],
                )?
            };

            let next_token_id = logits_processor.sample(&logits)?;
            if next_token_id as usize == builder.config.eos_token_id {
                break;
            }
            output_token_ids.push(next_token_id);
            if let Some(text) = tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                print!("{text}");
                result.push_str(&text);
                std::io::stdout().flush()?;
            }
        }
        let dt = start.elapsed();
        println!(
            "\n{} tokens generated ({:.2} token/s)\n",
            output_token_ids.len(),
            output_token_ids.len() as f64 / dt.as_secs_f64(),
        );
        Ok(result)
    }

    // build our application with a single route
    let app = Router::new().route("/completions", axum::routing::post(move |Json(payload): Json<serde_json::Value>| {
        let args = args.clone();
        async move {
            let builder = Arc::clone(&builder);
            let device = device.clone();
            let model = Arc::clone(&model);

             // Extract prompt from the payload
            let prompt = match payload.get("prompt") { Some(Value::String(s)) => s.clone(), _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    "Missing or invalid 'prompt' in request body".to_string(),
                ).into_response();
                }
            };

            match generate_text(&args, prompt, builder, &mut tokenizer, &device, model) {
                Ok(result) => {
                    let response = serde_json::json!({ "content": result });
                    axum::Json(response).into_response()
                },
                Err(err) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
            }
        }
    }));

    // run our app with hyper, listening globally on port 3000
    let port     = std::env::var("PORT").unwrap_or_else(|_| "10201".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}
