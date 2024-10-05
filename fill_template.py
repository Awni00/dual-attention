from jinja2 import Template

output_html_path= "index.html"
input_template_path = "index_template.html"
fig_dir = "./static/figs"

plotly_jinja_data = dict()

with open(f'{fig_dir}/relgames_learning_curves.html', 'r') as f:
    plotly_jinja_data['relfig'] = f.read()

with open(f'{fig_dir}/math_accuracy_scaling.html', 'r') as f:
    plotly_jinja_data['mathfig'] = f.read()

with open(f'{fig_dir}/language_modeling_scaling_laws.html', 'r') as f:
    plotly_jinja_data['langfig'] = f.read()

with open(output_html_path, "w", encoding="utf-8") as output_file:
    with open(input_template_path, encoding='utf-8') as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(plotly_jinja_data))