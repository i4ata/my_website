SELECT 
    position, 
    zone, 
    name, 
    total_points as points,
    goal_difference, 
    all_matches_played as matches_played, 
    all_matches_won as wins, 
    all_matches_lost as losses
FROM 
    league_table AS l
    JOIN team AS t ON l.id=t.id;
