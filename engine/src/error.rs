#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the embedded net.bin is malformed")]
    Net,
    #[error("tensor shape mismatch in a dense layer")]
    Shape,
}
