use schiphol;

create table if not exists flights (
    id char(18) primary key,
    estimatedLandingTime datetime,
    actualLandingTime datetime,
    scheduleDateTime datetime,
    flightName varchar(7),
    flightNumber int,
    airlineCode int,
    flightDirection char(1),
    destination char(3)
);

create table if not exists airlines (
    nvls int primary key,
    iata char(2),
    icao char(3),
    publicName varchar(100)
);

create table if not exists destinations (
    country varchar(100),
    iata char(3),
    publicName varchar(100),
    city varchar(100),
    latitude real,
    longitude real
);

load data infile '/docker-entrypoint-initdb.d/airlines.csv'
into table airlines
fields terminated by ',' optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(iata,icao,nvls,publicName);

load data infile '/docker-entrypoint-initdb.d/destinations.csv'
into table destinations
fields terminated by ',' optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(country,iata,publicName,city,@latitude, @longitude)
SET
    latitude = NULLIF(@latitude, ''),
    longitude = NULLIF(@longitude, '');

select count(*) as total_airlines_added from airlines;
select count(*) as total_destinations_added from destinations;