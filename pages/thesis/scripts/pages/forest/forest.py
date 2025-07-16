from dash import register_page
from pages.thesis.scripts.pages.forest.layout import layout
import pages.thesis.scripts.pages.forest.callbacks  # Import callbacks to register them

register_page(__name__, path='/thesis/forest')
