from dash import register_page
from pages.thesis.scripts.pages.vars.layout import layout
import pages.thesis.scripts.pages.vars.callbacks  # Import callbacks to register them

register_page(__name__, path='/thesis/variables', order=13)
