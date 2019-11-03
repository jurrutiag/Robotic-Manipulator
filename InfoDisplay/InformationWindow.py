import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanelItem, TabbedPanel
from kivy.lang import Builder
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


class TabWithInfoC(TabbedPanelItem):
    def __init__(self, process, index, **kwargs):
        super(TabWithInfoC, self).__init__(**kwargs)

        self.tab = Builder.template('TabWithInfo', process=process, tab_name=f"Process {index}")
        self.tab.ids.toggle_button.bind(on_press=self.toggle_graph)
        self.process = process

        self.param_value_model = None
        self.show_params = False

    def updateTab(self, info):

        if "Defaults" in info.keys():
            self.param_value_model = info["Defaults"].items()

        else:
            distance, torque, velocity = info["best_multi_fitness"]

            self.tab.ids.model_label.text = str(info["model"][1])
            self.tab.ids.generation_label.text = str(info["generation"])
            self.tab.ids.fitness_label.text = str(info["best_fitness"])
            self.tab.ids.distance_label.text = str(distance)
            self.tab.ids.torque_label.text = str(torque)
            self.tab.ids.velocity_label.text = str(velocity)
            self.tab.ids.mean_fitness_label.text = str(info["mean_fitness"])
            self.tab.ids.time_label.text = "%.2f s" % info["time_elapsed"]

            self.tab.ids.parameters_layout.clear_widgets()

            if self.show_params and self.param_value_model is not None:
                self.tab.ids.parameters_layout.cols = 2
                for param, value in self.param_value_model:
                    self.tab.ids.parameters_layout.add_widget(Label(text=str(param)))
                    self.tab.ids.parameters_layout.add_widget(Label(text=str(value)))
                    self.tab.ids.parameters_layout.add_widget(Builder.template('SmallYSeparator'))
                    self.tab.ids.parameters_layout.add_widget(Builder.template('SmallYSeparator'))
            else:
                self.tab.ids.parameters_layout.cols = 1
                # plt.gcf()
                fitness_graph = FigureCanvasKivyAgg(info["fitness_graph"])
                individual_graph = FigureCanvasKivyAgg(info["individual_graph"])

                # info["fitness_graph"].set_size_inches(4, 3)
                # info["individual_graph"].set_size_inches(4, 3)

                self.tab.ids.parameters_layout.add_widget(fitness_graph)
                fitness_graph.draw()
                self.tab.ids.parameters_layout.add_widget(individual_graph)
                individual_graph.draw()

                plt.close(fitness_graph.figure)
                plt.close(individual_graph.figure)

    def toggle_graph(self, instance):
        self.show_params = not self.show_params



class MainGrid(GridLayout):

    def __init__(self, queue, **kwargs):
        super(MainGrid, self).__init__(**kwargs)

        self.queue = queue
        self.processes = []
        self.tabs = []

        self.cols = 1


        self.tabbedPanel = TabbedPanel()
        self.tabbedPanel.do_default_tab = False
        self.add_widget(self.tabbedPanel)

        Clock.schedule_interval(self.update_info, 0.1)

    def update_info(self, dt):

        if not self.queue.empty():
            processes = [tab.process for tab in self.tabs]
            element = self.queue.get()
            process, model = element["model"]

            if process not in processes:
                new_tab = TabWithInfoC(process, len(processes) + 1)

                self.tabs.append(new_tab)
                self.tabbedPanel.add_widget(new_tab.tab)
                self.tabbedPanel.switch_to(new_tab.tab)

            current_tab = list(filter(lambda x: x.process == process, self.tabs))[0]


            current_tab.updateTab(element)


class InformationWindow(App):

    def __init__(self, queue, **kwargs):
        super(InformationWindow, self).__init__(**kwargs)
        self.queue = queue

    def build(self):
        w_width = 1080
        w_height = 607

        Window.minimum_width = w_width
        Window.minimum_height = w_height
        Window.size = (w_width, w_height)
        return MainGrid(self.queue)


def runInfoDisplay(queue):
    info_window = InformationWindow(queue=queue)
    info_window.run()
