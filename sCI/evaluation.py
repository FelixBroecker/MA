#!/usr/bin/env python3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from csf import SelectedCI

 
class Evaluation:
    def __init__(self):
        self.sCI = SelectedCI()

    def plot_ci_coefficients(self, wavefunction_name, n_elec):
        """plot ci coefficients of wavefunction"""
        csf_coefficients, csfs, CI_coefficients, _ = self.sCI.read_AMOLQC_csfs(
            f"{wavefunction_name}.wf", n_elec
        )
        # sort CI coefficients from largest to smallest absolut value and respectively csfs and csf_coefficients
        CI_coefficients_abs = -np.abs(np.array(CI_coefficients))
        idx = CI_coefficients_abs.argsort()
    
        CI_coefficients = [abs(CI_coefficients[i]) for i in idx]
        csf_coefficients = [csf_coefficients[i] for i in idx]
        csfs = [csfs[i] for i in idx]
    
        # plot ci coefficients
        plt.rcParams["figure.figsize"] = (10, 6.5)
        plt.rcParams["font.size"] = 16
        plt.rcParams["lines.linewidth"] = 3
        # plt.rcParams['font.family'] = 'Avenir'
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rc("axes", titlesize=16)
        plt.rc("axes", labelsize=16)
        plt.rc("axes", linewidth=2)
        color = []
        ref_state = self.sCI.build_energy_lowest_detetminant(n_elec)
        # color ci points by respective excitation
        for csf in csfs:
            difference = 0
            for electron in csf[0]:
                if electron not in ref_state:
                    difference += 1
            if difference == 1:
                color.append("darkred")
            elif difference == 2:
                color.append("grey")
            elif difference == 3:
                color.append("teal")
            elif difference == 4:
                color.append("orange")
            elif difference == 5:
                color.append("blue")
            elif difference == 6:
                color.append("lime")
            else:
                color.append("fuchsia")
    
        plt.xlabel("determinant idx [ ]", labelpad=15)
        plt.ylabel("CI coefficients [ ]", labelpad=15)
        plt.scatter(
            range(len(CI_coefficients) - 1),
            CI_coefficients[1:],
            s=15,
            marker=".",
            color=color[1:],
            label="CI coefficients",
        )
        plt.legend(
            loc="upper right",
        )
        plt.savefig(
            f"sCI_{wavefunction_name}.png",
            format="png",
            bbox_inches="tight",
            dpi=450,
        )

    def plot_ci_coefficients_plotly(self, wavefunction_name, n_elec, n_MO):
        """"""
        #load data
        csf_coefficients, csfs, CI_coefficients, _ = self.sCI.read_AMOLQC_csfs(
            f"{wavefunction_name}.wf", n_elec
        )
        # sort CI coefficients from largest to smallest absolut value and respectively csfs and csf_coefficients
        CI_coefficients_abs = -np.abs(np.array(CI_coefficients))
        idx = CI_coefficients_abs.argsort()
    
        CI_coefficients = [abs(CI_coefficients[i]) for i in idx]
        csf_coefficients = [csf_coefficients[i] for i in idx]
        csfs = [csfs[i] for i in idx]
        
        color = []
        ref_state = self.sCI.build_energy_lowest_detetminant(n_elec)
        
        for csf in csfs:
            difference = 0
            for electron in csf[0]:
                if electron not in ref_state:
                    difference += 1
            if difference == 1:
                color.append("darkred")
            elif difference == 2:
                color.append("grey")
            elif difference == 3:
                color.append("teal")
            elif difference == 4:
                color.append("orange")
            elif difference == 5:
                color.append("blue")
            elif difference == 6:
                color.append("lime")
            else:
                color.append("fuchsia")
        
        # get string representation of excitation
        all_string_representations = []
        for csf in csfs:
            determinant_alpha = ["-" for _ in range(n_MO)]
            determinant_beta = ["-" for _ in range(n_MO)]
            for elec in csf[0]:
                if elec > 0:
                    determinant_alpha[abs(elec)-1] = "+" 
                if elec < 0:
                    determinant_beta[abs(elec)-1] = "+" 
            #print(determinant_alpha)
            #print(determinant_beta)
            string_representation = ""
            for occ in determinant_alpha:
                string_representation += occ
            string_representation += "<br>"
            for occ in determinant_beta:
                string_representation += occ
            all_string_representations.append(string_representation)
        #print(all_string_representations)
        
        # Coordinates of the point
        x = np.arange(len(CI_coefficients[1:]))
        y = np.array(CI_coefficients[1:])
        
        # Create a scatter plot with hover data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, 
            y=y,
            mode='markers',
            marker=dict(
                size=4, 
                color=color[1:]
                ),
            hovertext=all_string_representations[1:],
        ))
        
        # Update layout
        fig.update_layout(
            width=800,   # Width of the plot (in pixels)
            height=600, 
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=False,       
                showline=True,        
                linecolor="black",    
                linewidth=2,          
                mirror=True,     
                ticklen=10,           
                ticks="outside",      
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                showgrid=False,       
                showline=True,        
                linecolor="black",    
                linewidth=2,          
                mirror=True,          
                ticklen=10,           
                ticks="outside",      
                tickfont=dict(size=14),  # Font size of ticks
            ),
            hoverlabel=dict(
                font=dict(
                    family="Courier New, monospace",  # Font family
                    size=16,         # Font size
                    color="black"    # Font color
                ),
                bgcolor="white",      # Background color of hover box
                bordercolor="black"    # Border color of hover box
            )
        )
        
        fig.write_html(f"{wavefunction_name}.html")
        
        fig.show()
