from dash import register_page
from pages.thesis.scripts.pages.most_common_tree.layout import layout
import pages.thesis.scripts.pages.most_common_tree.callbacks  # Import callbacks to register them

register_page(__name__, path='/thesis/most_common_tree', order=14)
