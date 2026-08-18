use std::env;
use std::path::PathBuf;

use epistemos_instant_recall::RecallEngine;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let root = take_option(&mut args, "--index")
        .map(PathBuf::from)
        .unwrap_or_else(default_index);
    let mut engine = RecallEngine::open(&root)?;
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "init" => println!("{}", root.display()),
        "index" => {
            let path = args
                .get(1)
                .ok_or("usage: recall index <path> [--index <directory>]")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.index_path(path)?)?
            );
        }
        "search" => {
            let query = args
                .get(1..)
                .ok_or("usage: recall search <query>")?
                .join(" ");
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.search_sidebar(&query, 10)?)?
            );
        }
        "recall" => {
            let text = args
                .get(1..)
                .ok_or("usage: recall recall <current text>")?
                .join(" ");
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.recall_ambient(None, &text, 5)?)?
            );
        }
        "stats" => println!(
            "{{\"documents\":{},\"index\":{}}}",
            engine.document_count(),
            serde_json::to_string(&root)?
        ),
        _ => print_help(),
    }
    Ok(())
}

fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let position = args.iter().position(|value| value == name)?;
    args.remove(position);
    if position < args.len() {
        Some(args.remove(position))
    } else {
        None
    }
}

fn default_index() -> PathBuf {
    env::var_os("EPISTEMOS_RECALL_INDEX")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("epistemos-instant-recall"))
}

fn print_help() {
    println!(
        "epistemos-instant-recall\n\n  init\n  index <file-or-directory>\n  search <query>\n  recall <current-note-text>\n  stats\n\nOptions:\n  --index <directory>"
    );
}
