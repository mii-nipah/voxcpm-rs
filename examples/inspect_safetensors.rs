use voxcpm_rs::tokenizer::TextTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TextTokenizer::from_local("/home/nipah/dev/ai_space/OmniVoice")?;
    
    let ids = tokenizer.encode("<|text_start|>Hello from OmniVoice")?;
    println!("'<|text_start|>Hello from OmniVoice' -> {:?}", ids);

    let ids2 = tokenizer.encode("<|text_start|>Hello")?;
    println!("'<|text_start|>Hello' -> {:?}", ids2);

    let ids3 = tokenizer.encode("Hello from OmniVoice")?;
    println!("'Hello from OmniVoice' -> {:?}", ids3);

    Ok(())
}
