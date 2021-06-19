use ndarray::prelude::*;
use std::fmt::Debug;
use std::vec::Vec;
use std::collections::VecDeque;

// TODO:
// 1/ min/max bounds for number of islands
// 2/ order rows and cols by priorities

enum Axis {
    ROW,
    COL
}

enum NodeType {
    ROOT,
    REGULAR
}

pub struct Problem {
    pub rows: Vec<Vec<usize>>,
    pub cols: Vec<Vec<usize>>,
    n_filled_in_row: Vec<usize>,
    n_filled_in_col: Vec<usize>
}

impl Problem {
    pub fn new(rows: Vec<Vec<usize>>, cols: Vec<Vec<usize>>) -> Problem {
        let mut n_filled_in_row = Vec::new();
        let mut n_filled_in_col = Vec::new();

        for row in &rows {
            n_filled_in_row.push(row.iter().sum());
        }

        for col in &cols {
            n_filled_in_col.push(col.iter().sum());
        }

        Problem{
            rows,
            cols,
            n_filled_in_row,
            n_filled_in_col
        }
    }
}

pub struct Board {
    board: Array2::<i8>,
    n_unknown: usize
}

impl Board {
    pub fn new(n: usize, m:usize) -> Self {
        Self{
            board: Array2::<i8>::zeros((n, m)),
            n_unknown: n * m
        }
    }

    fn set(&mut self, value: i8, axis_index: usize, ortho_index: usize, axis: &Axis) {
        match axis {
            Axis::ROW => {self.board[[ortho_index, axis_index]] = value;},
            Axis::COL => {self.board[[axis_index, ortho_index]] = value;}
        }
    }

    fn get(&mut self, axis_index: usize, ortho_index: usize, axis: &Axis) -> i8 {
        match axis {
            Axis::ROW => {self.board[[ortho_index, axis_index]]},
            Axis::COL => {self.board[[axis_index, ortho_index]]}
        }
    }
}

#[derive(PartialEq, Debug)]
struct Features {
    n_filled: usize,
    islands: Vec<usize>,
    n_unknown: usize,
}

impl Features {
    fn new(n_filled: usize, islands: Vec<usize>, n_unknown:usize) -> Features {
        Self{
            n_filled,
            islands,
            n_unknown
        }
    }
}

struct Node {
    node_type: NodeType,
    
    state: Vec<i8>,
    axis_index: usize, // index of expanded element in axis direction
    ortho_state: Vec<i8>,
    
    axis_features: Features,
    ortho_features: Features
}

fn print_node(node: &Node) {
    println!("state:{:?} islands:{:?} n_filled:{} n_unknown:{}, n_ortho_islands:{:?} n_ortho_filled:{} n_ortho_unknown:{}", node.state, node.axis_features.islands, node.axis_features.n_filled, node.axis_features.n_unknown,
        node.ortho_features.islands, node.ortho_features.n_filled, node.ortho_features.n_unknown);
}

fn get_next_to_expand(state: &Vec<i8>, start_index: usize) -> usize {
    for i in start_index..state.len() {
        if state[i] == 0 {
            return i
        }
    }
    state.len()
}

fn get_features(state: &Vec<i8>) -> Features {
    let mut n_filled = 0;
    let mut islands = Vec::with_capacity(state.len() / 2);
    let mut n_unknown = 0;
    
    let mut last = -1;

    for i in 0..state.len() {
        match (last, state[i]) {
            (0, 0) => { n_unknown+=1; },
            (0, 1) => { n_filled+=1; islands.push(1); last = state[i]},
            (0, -1) => { last = state[i]; },
            (1, 0) => { n_unknown+=1;},
            (1, 1) => { n_filled +=1; *islands.last_mut().unwrap()+=1; last = state[i];},
            (1, -1) => { last = state[i]; },
            (-1, 1) => { n_filled+=1; islands.push(1); last = state[i]; },
            (-1, 0) => { n_unknown+=1;},
            (-1, -1) => { last = state[i];},
            (_, _) => {}
        }

        last = state[i];
    }

    Features{
        n_filled, islands, n_unknown}
}

fn get_orthogonal_axis<'a>(axis: &Axis) -> Axis {
    match axis {
        Axis::ROW => Axis::COL,
        Axis::COL => Axis::ROW
    }   
}

fn get_axis_constraints<'a>(pb: &'a Problem, index: usize, axis: &Axis) -> &'a Vec<usize> {
    match axis {
        Axis::ROW => &pb.rows[index],
        Axis::COL => &pb.cols[index]
    }   
}

fn get_expected_n_filled<'a>(pb: &'a Problem, index: usize, axis: &Axis) -> usize {
    match axis {
        Axis::ROW => pb.n_filled_in_row[index],
        Axis::COL => pb.n_filled_in_col[index]
    }  
}

/*
fn get_expected_n_islands<'a>(pb: &'a Problem, index: usize, axis: &Axis) -> usize {
    match axis {
        Axis::ROW => pb.rows[index].len(),
        Axis::COL => pb.cols[index].len()
    }  
}*/

fn get_axis<'a>(board: &Array2::<i8>, index: usize, axis: &Axis) -> Vec<i8> {
    match axis {
        Axis::ROW => board.index_axis(Axis(0), index).to_vec(),
        Axis::COL => board.index_axis(Axis(1), index).to_vec(),
    }
}

fn get_axis_size<'a>(pb: &Problem, axis: &Axis) -> usize {
    match axis {
        Axis::ROW => pb.cols.len(),
        Axis::COL => pb.rows.len(),
    }
}

fn add_hypothesis_if_admissible<'a>(hypothesis: i8, node: &'a Node, pb: &'a Problem, board: &Array2::<i8>, axis_index: usize, ortho_index: usize, axis: &Axis, successors: &mut Vec::<Node>) {
    // First generate hypothesis in axis direction
    let mut state = node.state.clone();
    state[axis_index] = hypothesis;
    let axis_features = get_features(&state);

    if is_admissible(&axis_features, get_axis_constraints(pb, ortho_index, axis), get_expected_n_filled(pb, ortho_index, axis)) {
        // Second, if valid, generate hypothesis in ortho direction
        let ortho_axis = &get_orthogonal_axis(axis);
        let mut ortho_state = get_axis(board, axis_index, &get_orthogonal_axis(axis));
        ortho_state[ortho_index] = hypothesis;
        let ortho_features = get_features(&ortho_state);

        if is_admissible(&ortho_features, get_axis_constraints(pb, axis_index, ortho_axis), get_expected_n_filled(pb, axis_index, ortho_axis)) {
            // Add to successors if all constraints fullfilled
            successors.push(
                Node{
                    node_type: NodeType::REGULAR,
                    state,
                    axis_index,
                    ortho_state,
                    axis_features,
                    ortho_features}
                );
        }
    }
}

fn expand<'a>(node: &'a Node, pb: &'a Problem, board: &Array2::<i8>, ortho_index: usize, axis: &Axis) -> Vec::<Node> {
    let axis_index = match node.node_type {
        NodeType::ROOT => { get_next_to_expand(&node.state, 0) },
        _ => { get_next_to_expand(&node.state, node.axis_index + 1) }
    };

    let mut successors = Vec::<Node>::new();
     
    // empty hypothesis
    add_hypothesis_if_admissible(-1, node, pb, board, axis_index, ortho_index, axis, &mut successors);
    add_hypothesis_if_admissible(1, node, pb, board, axis_index, ortho_index, axis, &mut successors);
    
    successors
}

fn is_admissible(features: &Features, constraints: &Vec<usize>, expected_n_filled: usize) -> bool {
    match features.n_unknown {
        0 => features.islands == *constraints,
        _ => {
            // heuristic to prune early on bad candidates
            features.n_filled <= expected_n_filled && features.n_filled + features.n_unknown >= expected_n_filled// && features.islands.len() + features.n_unknown / 2 + 1 >= constraints.len()
        }
    }
}

fn generate_candidates<'a>(pb: &'a Problem, board: &Array2::<i8>, index: usize, axis: &Axis) -> Vec<Node> {
    let state = get_axis(board, index, axis);
    let ortho_state = get_axis(board, 0, &get_orthogonal_axis(axis));
    let axis_features = get_features(&state);
    let ortho_features = get_features(&ortho_state);
    let node = Node{
        node_type: NodeType::ROOT,
        state,
        axis_index : 0,
        ortho_state,
        axis_features,
        ortho_features
    };

    // generate hypothesis
    let mut stack = VecDeque::new();
    stack.push_back(node);

    let mut candidates = Vec::new();

    while !stack.is_empty() {
        let node = stack.pop_front().unwrap();

        if node.axis_features.n_unknown == 0 {
            candidates.push(node);
        }
        else {
            stack.extend(expand(&node, pb, board, index, axis));
        }
    }

    candidates  
}

fn step_on_axis<'a>(pb: &'a Problem, board: &mut Board, ortho_index: usize, axis: &Axis) {
    // generate hypothesis
    let candidates = generate_candidates(pb, &board.board, ortho_index, axis);

    // candidates
    //for node in &candidates {
    //    print!("candidate:{:?} ", node.state);
    //}

    for axis_index in 0..get_axis_size(pb, axis) {
        if board.get(axis_index, ortho_index, axis) == 0 && !candidates.is_empty() {
            let mut consensus = true;
            for i in 1..candidates.len() {
                consensus = consensus && (candidates[i-1].state[axis_index] == candidates[i].state[axis_index])
            }

            if consensus {
                let witness = candidates.first().unwrap();
                let value = witness.state[axis_index];

                board.set(value, axis_index, ortho_index, axis);
                board.n_unknown -= 1;
                //println!("board.n_unknown:{}", board.n_unknown);
            }
        }
    }

    //println!("board:\n{:?}", board);
}

fn step_on_board<'a>(pb: &'a Problem, board: &mut Board) {
    for i in 0..board.board.nrows() {
        step_on_axis(pb, board, i, &Axis::ROW);
    }

    for j in 0..board.board.ncols() {
        step_on_axis(pb, board, j, &Axis::COL);
    }
}

pub fn solve<'a>(pb: &'a Problem, board: &mut Board) {
    while board.n_unknown > 0 {
        step_on_board(pb, board);
    }
}

pub fn print_board(board: &Board) {
    for i in 0..board.board.nrows() {
        for j in 0..board.board.ncols() {
            print!("{}", if board.board[[i, j]] == 1 { 35 as char } else { 32 as char });
        }
        print!("\n");
    }

    print!("  ");
}

#[cfg(test)]
mod tests {

use crate::problems::*;
use super::*;

#[test]
fn test_get_features() {
    assert_eq!(get_features(&vec![-1, 1]), Features::new(1, vec![1], 0));
    assert_eq!(get_features(&vec![1, -1]), Features::new(1, vec![1], 0));
    assert_eq!(get_features(&vec![1, 1]), Features::new(2, vec![2], 0));
    assert_eq!(get_features(&vec![-1, 0]), Features::new(0, vec![], 1));
    assert_eq!(get_features(&vec![0, 0, 0, 0, 0]), Features::new(0, vec![], 5));
    assert_eq!(get_features(&vec![1, 0, 0, 0, 0]), Features::new(1, vec![1], 4));
    assert_eq!(get_features(&vec![1, -1, 1, 0, 0]), Features::new(2, vec![1, 1], 2));
    assert_eq!(get_features(&vec![-1, -1, -1, -1, -1]), Features::new(0, vec![], 0));
    assert_eq!(get_features(&vec![1, 1, 0, 1, 1]), Features::new(4, vec![2, 2], 1));
    assert_eq!(get_features(&vec![0, 0, 0, 0, 1]), Features::new(1, vec![1], 4));
    assert_eq!(get_features(&vec![1, -1, -1, -1, 1]), Features::new(2, vec![1, 1], 0));
    assert_eq!(get_features(&vec![1, -1, 1, -1, -1]), Features::new(2, vec![1, 1], 0));
}

#[test]
fn test_problem_10101() {
    let pb = create_problem_10101();

    let board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    let candidates = generate_candidates(&pb, &board, 0, &Axis::ROW);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates.first().unwrap().state, vec![1, -1, 1, -1, 1]);
}

#[test]
fn test_problem_11111() {
    let pb = create_problem_11111();

    let board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    let candidates = generate_candidates(&pb, &board, 0, &Axis::ROW);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates.first().unwrap().state, vec![1, 1, 1, 1, 1]);
}

#[test]
fn test_problem_110001_with_prefilled_board() {
    let pb = create_problem_110001();

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));
    board[[0, 0]] = 1;
    board[[0, 1]] = 1;
    board[[0, 2]] = -1;
    board[[0, 3]] = -1;
    board[[0, 4]] = 0;
    board[[0, 5]] = 1;

    let candidates = generate_candidates(&pb, &board, 0, &Axis::ROW);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates.first().unwrap().state, vec![1, 1, -1, -1, -1, 1]);
}

#[test]
fn test_problem_10100_11111() {
    let pb = create_problem_10100_11111();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());
  
    // solve
    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_simple_problem() {
    let pb = create_simple_problem();

    let mut board = Board::new(pb.rows.len(), pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_simple_6x6_problem() {
    let pb = create_simple_6x6_problem();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_medium_problem() {
    let pb = create_medium_problem();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_medium_10x10_problem() {
    let pb = create_medium_10x10_problem();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_2_29() {
    let pb = def_2_29();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_4_1() {
    let pb = def_4_1();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}

/*
#[test]
fn test_solve_5_392() {
    let pb = def_5_392();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board.board);
}*/

}