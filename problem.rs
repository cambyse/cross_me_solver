use ndarray::Array2;
use std::vec::Vec;

struct Problem {
    board: Array2<i8>,
    rows: Vec<Vec<i8>>,
    cols: Vec<Vec<i8>>,
}