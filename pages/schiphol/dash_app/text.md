# Schiphol Airport

In this project I develop a simple ETL pipeline that enables for interaction and analytics of the flights to and from Amsterdam's Schiphol Airport (4th busiest airport in Europe, 3rd most connected airport in the world). The flight information is sourced daily directly from the public [API](https://developer.schiphol.nl/) provided directly by the airport. Minimal preprocessing and cleaning is done with Pandas, after which the data is loaded into a MySQL database. Then, the following Plotly Dash application visualizes the database. Everything is orchestrated automatically using Apache Airflow. This project is still in development!

![Schiphol](../../../assets/schiphol/Schiphol.svg)

The Entity Relation diagram of the database is the following:

![ER](../../../assets/schiphol/ER.svg)

Here the `flights` table is populated once a day at 03:00 with the flight information for today. The other tables remain constant and are also sourced from the same API. The coordinates for each airport are sourced from [here](https://data.opendatasoft.com/explore/dataset/airports-code%40public/export/?dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6ImFpcnBvcnRzLWNvZGVAcHVibGljIiwib3B0aW9ucyI6e319LCJjaGFydHMiOlt7ImFsaWduTW9udGgiOnRydWUsInR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiQVZHIiwieUF4aXMiOiJsYXRpdHVkZSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiMxNDJFN0IifV0sInhBeGlzIjoiY291bnRyeV9uYW1lIiwibWF4cG9pbnRzIjo1MCwic29ydCI6IiJ9XSwidGltZXNjYWxlIjoiIiwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZX0%3D&location=2,41.08309,0.07266&basemap=jawg.streets).
