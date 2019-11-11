import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanelItem, TabbedPanel
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import ast
import json
from kivy.factory import Factory
import os


class TabWithInfo(TabbedPanelItem):
    def __init__(self, process, **kwargs):
        super(TabWithInfo, self).__init__(**kwargs)

        self.process = process

        self.last_graphs = []
        self.param_value_model = None

    def updateTab(self, info):

        if "Defaults" in info.keys():
            self.param_value_model = info["Defaults"].items()
            self.showParams()

        else:
            distance, torque, velocity = info["best_multi_fitness"]
            status = " (Running)" if not info["terminate"] else " (Finished)"

            self.ids.model_label.text = str(info["model"][1]) + status
            self.ids.generation_label.text = str(info["generation"])
            self.ids.fitness_label.text = str(info["best_fitness"])
            self.ids.distance_label.text = str(distance)
            self.ids.torque_label.text = str(torque)
            self.ids.velocity_label.text = str(velocity)
            self.ids.mean_fitness_label.text = str(info["mean_fitness"])
            self.ids.time_label.text = "%.2f s" % info["time_elapsed"]

            self.last_graphs = [info["fitness_graph"], info["individual_graph"], info["pareto_graph"]]
            self.showGraphs()

    def showParams(self):
        self.ids.params_tab.clear_widgets()

        for param, value in sorted(self.param_value_model):
            self.ids.params_tab.add_widget(Label(text=str(param)))
            self.ids.params_tab.add_widget(Label(text=str(value)))
            self.ids.params_tab.add_widget(Factory.SmallYSeparator())
            self.ids.params_tab.add_widget(Factory.SmallYSeparator())

    def showGraphs(self):
        if self.last_graphs is not None:
            self.ids.general_graphs.clear_widgets()
            self.ids.pareto_graph.clear_widgets()

            fitness_graph = FigureCanvasKivyAgg(self.last_graphs[0])
            individual_graph = FigureCanvasKivyAgg(self.last_graphs[1])
            pareto_graph = FigureCanvasKivyAgg(self.last_graphs[2])

            self.ids.general_graphs.add_widget(fitness_graph)
            fitness_graph.draw()
            self.ids.general_graphs.add_widget(individual_graph)
            individual_graph.draw()
            self.ids.pareto_graph.add_widget(pareto_graph)
            pareto_graph.draw()


class MainGrid(GridLayout):

    def __init__(self, queue, **kwargs):
        super(MainGrid, self).__init__(**kwargs)

        self.queue = queue
        self.cores = []
        self.tabs = []

        self.cols = 1

        self.tabbedPanel = TabbedPanel()
        self.tabbedPanel.do_default_tab = False
        self.add_widget(self.tabbedPanel)

        Clock.schedule_interval(self.update_info, 0.1)

    def update_info(self, dt):

        if not self.queue.empty():
            cores = [tab.process for tab in self.tabs]
            element = self.queue.get()
            process, model = element["model"]

            if process not in cores:
                new_tab = TabWithInfo(process)
                new_tab.text = f"Process {len(cores) + 1}"

                self.tabs.append(new_tab)
                self.tabbedPanel.add_widget(new_tab)
                self.tabbedPanel.switch_to(new_tab)

            current_tab = list(filter(lambda x: x.process == process, self.tabs))[0]

            current_tab.updateTab(element)


class MainWindowGrid(GridLayout):

    pass


class InformationWindow(App):

    def __init__(self, queue, title, **kwargs):
        super(InformationWindow, self).__init__(**kwargs)
        self.queue = queue
        self.title = 'Model: ' + title

    def build(self):
        w_width = 1080
        w_height = 607

        Window.minimum_width = w_width
        Window.minimum_height = w_height
        Window.size = (w_width, w_height)

        return MainGrid(self.queue)


class MainWindow(App):
    def __init__(self, default_params, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        # Run
        self.chosen_option = None
        self.default_params = default_params
        self.text_inputs = {}
        self.use_defaults = False
        self.run_window = None
        self.cores = 1

        # Render
        self.render_window = None
        self.individuals_checkboxes = []
        self.all_runs_render = False
        self.render_model_name = ''
        self.render_run = 0

        self.information = {}

        for key, value in self.default_params.items():
            self.default_params[key] = [value]

        with open('../InfoDisplay/parameters_variations.json', 'r') as f:
            self.parameters_variations = json.load(f)

    def build(self):
        self.title = 'Robotic Manipulator'
        return MainWindowGrid()

    def runAll(self):
        self.chosen_option = 1
        self.root.ids.run_button.text = "Run (Run All)"
        self.root.ids.info_layout.clear_widgets()
        self.runWindow()

    def initializeOnly(self):
        self.chosen_option = 2
        self.root.ids.run_button.text = "Run (Initialize Only)"
        self.root.ids.info_layout.clear_widgets()

    def profiling(self):
        self.chosen_option = 3
        self.root.ids.run_button.text = "Run (Profiling)"
        self.root.ids.info_layout.clear_widgets()

    def render(self):
        self.chosen_option = 4
        self.root.ids.run_button.text = "Run (Render)"
        self.root.ids.info_layout.clear_widgets()
        self.renderWindow()

    def findParetoFrontier(self):
        self.chosen_option = 5
        self.root.ids.run_button.text = "Run (Find Pareto Frontier)"
        self.root.ids.info_layout.clear_widgets()

    def runButton(self):
        if self.chosen_option == 1:
            self.parameters_variations = {key: ast.literal_eval('[' + t_input.text + ']') for key, t_input in self.text_inputs.items()}

            with open('../InfoDisplay/parameters_variations.json', 'w') as f:
                json.dump(self.parameters_variations, f)

            try:
                self.cores = int(self.run_window.ids.cores.text)
            except ValueError:
                self.cores = 1

            self.information = {
                'parameters_variations': self.parameters_variations,
                'cores': self.cores,
                'run_name': self.run_window.ids.run_name.text
            }
        elif self.chosen_option == 4:
            self.information = {
                'render_model_name': self.render_model_name,
                'render_run': self.render_run,
                'render_individuals': [i for i, val in enumerate([chbox.active for chbox in self.individuals_checkboxes]) if val]
            }

        self.information['final_option'] = self.chosen_option

        self.stop()

    def on_checkbox_active(self, checkbox, value):
        self.use_defaults = value
        self.showParameters()

    def runWindow(self):
        self.run_window = Factory.RunWindow()
        self.root.ids.info_layout.add_widget(self.run_window)
        self.use_defaults = self.run_window.ids.use_defaults.active
        self.showParameters()

    def showParameters(self):
        self.run_window.ids.parameters_layout.clear_widgets()
        the_params = self.default_params if self.use_defaults else self.parameters_variations

        for param, value in sorted(the_params.items()):
            self.text_inputs[param] = TextInput(text=', '.join(map(str, value)), multiline=False)
            self.run_window.ids.parameters_layout.add_widget(Label(text=str(param)))
            self.run_window.ids.parameters_layout.add_widget(self.text_inputs[param])

    def renderWindow(self):
        self.render_window = Factory.RenderWindow()
        self.root.ids.info_layout.add_widget(self.render_window)
        models = [name for name in os.listdir("../Model/Trained Models") if os.path.isdir(f'../Model/Trained Models/{name}')]

        self.render_window.ids.model_selection.values = models

    def selectedModel(self, instance, model):
        self.render_model_name = model

        if not self.all_runs_render:
            self.render_window.ids.individuals_selection.clear_widgets()
            amount_of_runs = len([name for name in os.listdir(f"../Model/Trained Models/{model}/Graphs/Individuals") if os.path.isdir(f'../Model/Trained Models/{model}/Graphs/Individuals/{name}')])

            self.render_window.ids.run_selection.values = list(map(str, sorted(range(amount_of_runs), reverse=True)))

    def selectedRun(self, instance, run):
        self.render_run = int(run)

        self.render_window.ids.individuals_selection.clear_widgets()
        amount_of_individuals = len([name for name in os.listdir(f"../Model/Trained Models/{self.render_model_name}/Graphs/Individuals/{run}") if os.path.isfile(f'../Model/Trained Models/{self.render_model_name}/Graphs/Individuals/{run}/{name}')])

        self.individuals_checkboxes = []
        for i in range(amount_of_individuals):
            self.individuals_checkboxes.append(CheckBox())
            self.render_window.ids.individuals_selection.add_widget(Label(text=f'Individual {i}'))
            self.render_window.ids.individuals_selection.add_widget(self.individuals_checkboxes[i])

    def selectAllIndividuals(self):
        if all([chbox.active for chbox in self.individuals_checkboxes]):
            for chbox in self.individuals_checkboxes:
                chbox.active = False
        else:
            for chbox in self.individuals_checkboxes:
                chbox.active = True

    def allRunsCheckbox(self, checkbox, value):
        self.all_runs_render = value
        self.render_window.ids.render_selections.size_hint_x = 0 if value else 1
        self.render_window.ids.render_selections.opacity = 0 if value else 1
        self.render_window.ids.render_selections.disabled = value
        self.render_window.ids.render_selections.visible = not value


def runInfoDisplay(queue, title):
    info_window = InformationWindow(queue=queue, title=title)
    info_window.run()


def runMainWindow(default_params):
    main_window = MainWindow(default_params)
    main_window.run()

    main_window_information = main_window.information

    return main_window_information
