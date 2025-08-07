SELECT 
    date, 
    t1.name AS home_team, 
    t2.name AS away_team, 
    (home_team_score || ' - ' || away_team_score) AS score,
    (SIGN(f.home_team_score - f.away_team_score) * CASE WHEN f.home_team=3063 THEN 1 ELSE -1 END) AS res
FROM 
    fixture AS f 
    JOIN team AS t1 ON f.home_team=t1.id 
    JOIN team AS t2 ON f.away_team=t2.id
WHERE 
    FT=1 AND (home_team = 3063 OR away_team = 3063) 
LIMIT 5;
