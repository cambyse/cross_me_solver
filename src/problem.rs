use ndarray::prelude::*;
use std::fmt::Debug;
use std::vec::Vec;
use std::collections::VecDeque;

enum Axis {
    ROW,
    COL
}

enum NodeType {
    ROOT,
    REGULAR
}

struct Problem {
    rows: Vec<Vec<usize>>,
    cols: Vec<Vec<usize>>,
    n_filled_in_row: Vec<usize>,
    n_filled_in_col: Vec<usize>
}

impl Problem {
    fn new(rows: Vec<Vec<usize>>, cols: Vec<Vec<usize>>) -> Problem {
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

// Idea optimistic and pessimistic islands!

struct Node {
    node_type: NodeType,
    // expansion state
    axis_index: usize,
    state: Vec<i8>,
    ortho_index: usize,
    ortho_state: Vec<i8>,

    // features in axis dir
    axis_features: Features,

    // features in orthogonal dir
    ortho_features: Features
}

fn print_node(node: &Node) {
    println!("state:{:?} islands:{:?} n_filled:{} n_unknown:{}, n_ortho_islands:{:?} n_ortho_filled:{} n_ortho_unknown:{}", node.state, node.axis_features.islands, node.axis_features.n_filled, node.axis_features.n_unknown,
        node.ortho_features.islands, node.ortho_features.n_filled, node.ortho_features.n_unknown);
}

fn get_to_expand(state: &Vec<i8>, start_index: usize) -> usize {
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

fn get_axis_constraints<'a>(pb: &'a Problem, index: usize, axis: &Axis) -> Vec<usize> {
    match axis {
        Axis::ROW => pb.rows[index].clone(),
        Axis::COL => pb.cols[index].clone()
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

fn get<'a>(node: &'a Node, j:usize, axis: &Axis) -> i8 {
    match axis {
        Axis::ROW => node.state[j],
        Axis::COL => node.ortho_state[j],
    }
}

fn set<'a>(board: &mut Array2::<i8>, value: i8, index: usize, sub_index: usize, axis: &Axis) {
    match axis {
        Axis::ROW => {board[[index, sub_index]] = value;},
        Axis::COL => {board[[sub_index, index]] = value; }
    }
}

fn get_axis_size<'a>(pb: &Problem, axis: &Axis) -> usize {
    match axis {
        Axis::ROW => pb.cols.len(),
        Axis::COL => pb.rows.len(),
    }
}

fn expand<'a>(node: &'a Node, pb: &'a Problem, board: &Array2::<i8>, index: usize, axis: &Axis) -> Vec::<Node> {
    let ortho_index = match node.node_type {
        NodeType::ROOT => { get_to_expand(&node.state, 0) },
        _ => { get_to_expand(&node.state, node.ortho_index + 1) }
    };
    
    if node.axis_features.n_unknown == 0 {
        return vec![];
    }

    let mut successors = Vec::<Node>::new();
     
    // empty hypothesis

    let mut state = node.state.clone();
    state[ortho_index] = -1;
    let mut ortho_state = get_axis(board, ortho_index, &get_orthogonal_axis(axis));
    ortho_state[node.axis_index] = -1;
    let axis_features = get_features(&state);
    let ortho_features = get_features(&ortho_state);
    
    let empty_hypothesis = Node{
        node_type: NodeType::REGULAR,
        axis_index: node.axis_index,
        state,
        ortho_index,
        ortho_state,
        axis_features,
        ortho_features
    };

    //println!("Empty hypothesis:");
    //print_node(&empty_hypothesis);

    if is_admissible(&empty_hypothesis, pb, index, axis) {
        successors.push(empty_hypothesis);
    }


    // fill hypothesis
    let mut state = node.state.clone();
    state[ortho_index] = 1;
    let mut ortho_state = get_axis(board, ortho_index, &get_orthogonal_axis(axis));
    ortho_state[node.axis_index] = 1;
    let axis_features = get_features(&state);
    let ortho_features = get_features(&ortho_state);

    let filled_hypothesis = Node{
        node_type: NodeType::REGULAR,
        axis_index: node.axis_index,
        state,
        ortho_index,
        ortho_state,
        axis_features,
        ortho_features
    };

    //println!("Fill hypothesis:");
    //print_node(&filled_hypothesis);

    if is_admissible(&filled_hypothesis, pb, index, axis) {
        successors.push(filled_hypothesis);
    }

    successors
}

fn is_admissible<'a>(node: &'a Node, pb: &'a Problem, index: usize, axis: &Axis) -> bool {    
    // in axis drection
    let mut admissible_in_axis_direction = true;
    {
        //println!("\tlongitudinal..");

        let expected_n_filled = get_expected_n_filled(pb, index, axis); // do better bookkeeping!
        //let expected_n_islands = get_expected_n_islands(pb, index, axis);

        if node.axis_features.n_unknown == 0 {
            let constraints = get_axis_constraints(pb, index, axis);

            admissible_in_axis_direction = constraints == node.axis_features.islands;

            //println!("\t\tis_finished: {:?}", constraints);
        }
        else {
            //admissible_in_axis_direction = admissible_in_axis_direction && node.axis_features.n_islands <= expected_n_islands;
            admissible_in_axis_direction = admissible_in_axis_direction && node.axis_features.n_filled <= expected_n_filled;
            admissible_in_axis_direction = admissible_in_axis_direction && node.axis_features.n_filled + node.axis_features.n_unknown >= expected_n_filled;

            //println!("\t\tnode.n_islands <= expected_n_islands: {}", node.axis_features.n_islands <= expected_n_islands);
            //println!("\t\tnode.n_filled <= expected_n_filled: {}", node.axis_features.n_filled <= expected_n_filled);
            //println!("\t\tnode.n_filled + node.n_unknown >= expected_n_filled: {}", node.axis_features.n_filled + node.axis_features.n_unknown >= expected_n_filled);
        }
    }

    // in orthogonal direction
    let mut admissible_in_ortho_direction = true;

    {
        //println!("\torthogonal..");

        let axis = &get_orthogonal_axis(axis);

        let expected_n_ortho_filled = get_expected_n_filled(pb, node.ortho_index, axis); // do better bookkeeping!
        
        if node.ortho_features.n_unknown == 0 {
            let constraints = get_axis_constraints(pb, node.ortho_index, axis);

            admissible_in_ortho_direction = constraints == node.ortho_features.islands;

            //println!("\t\tis_ortho_finished: {:?}", constraints);
        }
        else {
            admissible_in_ortho_direction = admissible_in_ortho_direction && node.ortho_features.n_filled <= expected_n_ortho_filled;
            admissible_in_ortho_direction = admissible_in_ortho_direction && node.ortho_features.n_filled + node.ortho_features.n_unknown >= expected_n_ortho_filled;

            //println!("\t\tnode.n_filled <= expected_n_filled: {}", node.ortho_features.n_filled <= expected_n_ortho_filled);
            //println!("\t\tnode.n_filled + node.n_unknown >= expected_n_filled: {}", node.ortho_features.n_filled + node.ortho_features.n_unknown >= expected_n_ortho_filled);
        }
    }

    //println!("\t{}", admissible_in_axis_direction && admissible_in_ortho_direction);

    admissible_in_axis_direction && admissible_in_ortho_direction
}

fn generate_candidates<'a>(pb: &'a Problem, board: &Array2::<i8>, index: usize, axis: &Axis) -> Vec<Node> {
    let state = get_axis(board, index, axis);
    let ortho_state = get_axis(board, 0, &get_orthogonal_axis(axis));
    let axis_features = get_features(&state);
    let ortho_features = get_features(&ortho_state);
    let node = Node{
        node_type: NodeType::ROOT,
        axis_index: index,
        state,
        ortho_index : 0,
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

        //println!("POP");
        //print_node(&node);
        if node.axis_features.n_unknown == 0 {
            candidates.push(node);
        }
        else {
            let successors = expand(&node, pb, board, index, axis);
            stack.extend(successors);
        }
    }

    candidates  
}

fn step_on_line<'a>(pb: &'a Problem, board: &mut Array2::<i8>, index: usize, axis: &Axis) {
    // generate hypothesis
    let candidates = generate_candidates(pb, board, index, axis);

    // candidates
    //for node in &candidates {
    //    print!("candidate:{:?} ", node.state);
    //}

    for j in 0..get_axis_size(pb, axis) {
        let mut consensus = true;
        for i in 1..candidates.len() {
            consensus = consensus && (candidates[i-1].state[j] == candidates[i].state[j])
        }

        if consensus && !candidates.is_empty() { // TODO: don't set if already set?
            let witness = candidates.first().unwrap();
            let value = witness.state[j];

            //println!("set value:{} {} <- {}", index, j, value);

            set(board, value, index, j, axis)
        }
    }

    println!("board:\n{:?}", board);
}

fn step_on_board<'a>(pb: &'a Problem, board: &mut Array2::<i8>) {
    //println!("ROWS");
    for i in 0..board.nrows() {
        println!("row[{}]", i);
        step_on_line(pb, board, i, &Axis::ROW);
    }

    //println!("COLS");
    for j in 0..board.ncols() {
        println!("col[{}]", j);
        step_on_line(pb, board, j, &Axis::COL);
    }
}

fn solve<'a>(pb: &'a Problem, board: &mut Array2::<i8>) {
    while board.iter().any(|&e| e==0) {
        step_on_board(pb, board);
    }
}

fn print_board(board: &Array2::<i8>) {
    for i in 0..board.nrows() {
        for j in 0..board.ncols() {
            print!("{}", if board[[i, j]] == 1 { 35 as char } else { 32 as char });
        }
        print!("\n");
    }

    print!("  ");
}

#[cfg(test)]
mod tests {

use std::vec;

use super::*;

fn create_simple_problem() -> Problem {
    Problem::new( 
        vec![vec![3], vec![1, 1, 1], vec![3], vec![1, 1], vec![1, 1]],
        vec![vec![1, 1], vec![1, 2], vec![3], vec![1, 2], vec![1, 1]] )
}

fn create_simple_6x6_problem() -> Problem {
    Problem::new( 
        vec![vec![2,1], vec![1, 3], vec![1,2], vec![3], vec![4], vec![1]],
        vec![vec![1], vec![5], vec![2], vec![5], vec![2, 1], vec![2]])
}

fn create_medium_problem() -> Problem {
    Problem::new(
        vec![vec![3], vec![3, 1], vec![2, 1, 1], vec![4, 1], vec![3, 1, 1], vec![2, 1], vec![3]],
        vec![vec![3], vec![5], vec![2, 4], vec![4, 1], vec![1, 1, 1], vec![1, 1], vec![3]])
}

fn create_medium_10x10_problem() -> Problem {
    Problem::new(
        vec![vec![4], vec![1, 1], vec![1, 4, 1], vec![4, 4], vec![1, 5, 1], vec![4, 4], vec![3, 2, 3], vec![3, 2, 3], vec![4, 4], vec![10]],
        vec![vec![7], vec![2, 5], vec![1, 5], vec![4, 2], vec![1, 1, 2, 1], vec![3, 1, 2, 1], vec![1, 4, 2], vec![1, 7], vec![4, 5], vec![7]])
}

fn create_problem_10101() -> Problem {
    Problem::new(
        vec![vec![1, 1, 1]],
        vec![vec![1], vec![], vec![1], vec![], vec![1]])
}

fn create_problem_11111() -> Problem {
    Problem::new(
        vec![vec![5]],
        vec![vec![1], vec![1], vec![1], vec![1], vec![1]])
}

fn create_problem_110001() -> Problem {
    Problem::new(
        vec![vec![2, 1]],
        vec![vec![1], vec![1], vec![], vec![], vec![], vec![1]])
}

fn create_problem_10100_11111() -> Problem {
    Problem::new( 
        vec![vec![1, 1], vec![5]],
        vec![vec![2], vec![1], vec![2], vec![1], vec![1]])
}

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

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));
  
    // solve
    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_simple_problem() {
    let pb = create_simple_problem();

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_simple_6x6_problem() {
    let pb = create_simple_6x6_problem();

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_medium_problem() {
    let pb = create_medium_problem();

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    solve(&pb, &mut board);

    print_board(&board);
}

#[test]
fn test_solve_medium_10x10_problem() {
    let pb = create_medium_10x10_problem();

    let mut board = Array2::<i8>::zeros((pb.rows.len(), pb.cols.len()));

    solve(&pb, &mut board);

    print_board(&board);
}

}