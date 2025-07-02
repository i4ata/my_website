from dash import register_page
from pages.thesis.scripts.pages.home.layout import layout
import pages.thesis.scripts.pages.home.callbacks  # Import callbacks to register them

register_page(__name__, path='/thesis', order=11, name='Thesis')
