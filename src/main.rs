use cross_me_solver::solver::*;
use cross_me_solver::problems::*;

fn main() {
    let pb = def_4_1();

    let mut board = Board::new(pb.rows.len(),pb.cols.len());

    solve(&pb, &mut board);

    print_board(&board);
}
