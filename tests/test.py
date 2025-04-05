import dash
from dash import html

app = dash.Dash(__name__)

with open("TrackingQuickstart.html") as f:
    text = f.readlines()
app.layout = html.Div([
    html.H1('THIS WORKS'),
    html.Iframe(srcDoc=''.join(text), style={"height": "800px", "width": "100%"})
])

if __name__ == '__main__':
    app.run_server(debug=True)
