html_layout = """
<!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%css%}
        </head>
        <body class="dash-template">
            <header>
                <div class="nav-wrapper">
                    <a href="/">
                        <h1>Visualization Dashboard</h1>
                    </a>
                </div>
            </header>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""