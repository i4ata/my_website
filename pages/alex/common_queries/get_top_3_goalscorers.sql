SELECT 
    first_name,
    last_name,
    COUNT(*) AS goals 
FROM 
    goal AS g 
    JOIN player AS p ON g.player=p.id
WHERE team = 3063 
GROUP BY player 
ORDER BY goals DESC 
LIMIT 3;
