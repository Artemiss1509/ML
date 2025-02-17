def player(prev_play, opponent_history=[], sequences={}):
    stride = 3

    if prev_play:
        opponent_history.append(prev_play)

    # Opponent_history length capped at stride + 1
    while len(opponent_history) > stride + 1:
        opponent_history.pop(0)

    # Not enough history, default to "R"
    if len(opponent_history) <= stride:
        return "R"

    # Update the sequences dictionary
    current_sequence = "".join(opponent_history)
    sequences[current_sequence] = sequences.get(current_sequence, 0) + 1

    # Get the last 'stride' moves
    last_moves = "".join(opponent_history[-stride:])
    possible_next_moves = [last_moves + "R", last_moves + "P", last_moves + "S"]

    # Determine most frequent next move
    most_frequent = max(possible_next_moves, key=lambda k: sequences.get(k, 0))
    predicted_move = most_frequent[-1]

    if predicted_move == "R":
        return "P"
    elif predicted_move == "P":
        return "S"
    else:
        return "R"