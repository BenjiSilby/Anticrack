import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Storm labels
storms = ["Feb 7", "Feb 14*", "Feb 20", "Mar 4", "Mar 13", "Apr 1 AM", "Apr 1 PM"]

# Input data
data = {
    "Storm": storms,
    
    "Mixed-Mode Z_fracture": [5.0, 6.0, 16.0, 3.0, 2.0, 7.5, 21.0],
    "Mixed-Mode F_imp": [115.7, 115.7, 25.3, 97.6, 97.6, 61.4, 133.8],
    "Mixed-Mode HS_test": [19.0, 24.5, 21.5, 24.0, 17.0, 20.0, 31.0],
    "Mixed-Mode Angle": [20, 28, 25, 26, 23, 20, 20],

    "Mode II Z_fracture": [1.0, 26.5, 19.0, 23.0, 2.0, 22.0, 2.0],
    "Mode II HS_test": [15.0, 26.5, 19.0, 23.0, 18.0, 22.0, 30.0],
    "Mode II Angle": [90, 90, 90, 90, 70, 90, 55],

    "Mode I Z_fracture": [0.5, 11.5, 13.0, 4.0, 2.0, 7.5, 27.0],
    "Mode I F_imp": [169.9, 43.4, 97.6, 169.9, 133.8, 97.6, None],
    "Mode I HS_test": [18.5, 26.0, 20.0, 25.0, 19.5, 20.5, 27.0],
}

df = pd.DataFrame(data)
x = np.arange(len(df))
bar_width = 0.2

def plot_grouped_bar(ax, hs_data, z_data, f_data, angle_data, storm_labels, colors, test_label):
    # Define LaTeX-style labels
    hs_label = r"$HS_{test}$ (cm)"
    z_label = r"$Z_{fracture}$ (cm)"
    f_label = r"$F_{imp}$ (N)"
    angle_label = "Angle (°)"

    # Plot HS and Z on left axis
    ax.bar(x - bar_width, hs_data, width=bar_width, label=hs_label, color=colors[0], zorder = 1)
    ax.bar(x, z_data, width=bar_width, label=z_label, color=colors[1], zorder = 1)
    ax.set_ylabel("Height (cm)") 
    
    # Right Y-axis for F_imp and/or Angle
    if f_data is not None or angle_data is not None:
        ax2 = ax.twinx()
        if f_data is not None:
            ax2.bar(x + bar_width, f_data, width=bar_width, label=f_label, color=colors[2], zorder = 0.99)
        if angle_data is not None:
            ax2.plot(x + bar_width, angle_data, marker='o', linestyle='--', color=colors[3], label=angle_label, zorder = 0.99)
        ax2.set_ylabel("Angle")
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right').set_zorder(2)
    if f_data is not None and angle_data is not None:
        ax2.set_ylabel("Force / Angle")
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right').set_zorder(2)
    if f_data is not None and angle_data is None:
        ax2.set_ylabel("Force (N)")
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right').set_zorder(2)   

    # X-axis and layout
    ax.set_xticks(x)
    ax.set_xticklabels(storm_labels, rotation=45)
    ax.set_xlabel("Tiltboard Test Date (2025)")
    ax.set_title(test_label)
    ax.legend(loc='upper left').set_zorder(4)

    
    return ax, ax2



def make_plot(hs_col, z_col, f_col, angle_col, title, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax, ax2 = plot_grouped_bar(
        ax,
        df[hs_col],
        df[z_col],
        df[f_col] if f_col else None,
        df[angle_col] if angle_col else None,
        df["Storm"],
        colors,
        title
    )
    # Get legend handles and labels from ax
    handles1, labels1 = ax.get_legend_handles_labels()
    
    # Get legend handles and labels from ax2 if it exists
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []

    # Remove legends from axes (if any)
    ax.legend_.remove() if ax.legend_ else None
    if ax2 and ax2.legend_:
        ax2.legend_.remove()

    # Add first legend at upper left (figure coords)
    fig.legend(handles1, labels1, loc='upper left',
               bbox_to_anchor=(0.075, 0.825),
               frameon=True)

    # Add second legend at upper right (figure coords)
    if handles2:
        fig.legend(handles2, labels2, loc='upper right',
                   bbox_to_anchor=(0.925, 0.825),
                   frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.9])  # leave space for legends at top
    plt.show() 



# Plot all three test types
make_plot("Mixed-Mode HS_test", "Mixed-Mode Z_fracture", "Mixed-Mode F_imp", "Mixed-Mode Angle",
          "Mixed-Mode Tiltboard Tests", colors=["lightgray", "skyblue", "coral", "black"])

make_plot("Mode II HS_test", "Mode II Z_fracture", None, "Mode II Angle",
          "Mode II Tiltboard Tests", colors=["lightgray", "lightgreen", None, "black"])

make_plot("Mode I HS_test", "Mode I Z_fracture", "Mode I F_imp", None,
          "Mode I Tiltboard Tests", colors=["lightgray", "violet", "goldenrod", None])

# try another plot
fig, ax = plt.subplots(figsize=(10, 6), dpi = 400)
bars =  ax.bar(x - bar_width, df["Mode I HS_test"].values, width=bar_width, label=r"$HS_{test}$ (cm)", color = "lightgray", edgecolor='black')
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height+1, "Mode I", 
            ha='center', va='bottom', rotation = 90)
bars =  ax.bar(x - bar_width, df["Mode I Z_fracture"].values, width=bar_width, label=r"$Z_{fracture}$ (cm)", color = "coral", edgecolor='black')

bars =  ax.bar(x, df["Mode II HS_test"].values, width=bar_width, color = "lightgray", edgecolor='black')
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height+1, "Mode II", 
            ha='center', va='bottom', rotation = 90)
bars =  ax.bar(x , df["Mode II Z_fracture"].values, width=bar_width, color = "coral", edgecolor='black')

bars =  ax.bar(x + bar_width, df["Mixed-Mode HS_test"].values, width=bar_width, color = "lightgray", edgecolor='black')
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height+1, "Mixed-Mode", 
            ha='center', va='bottom', rotation = 90)
bars =  ax.bar(x + bar_width, df["Mixed-Mode Z_fracture"].values, width=bar_width, color = "coral", edgecolor='black')
plt.legend(loc="upper left")
plt.yticks(np.arange(0, 41, 2))  # tighter tick spacing
plt.xticks(x, storms, rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
plt.ylim(top = 40)
plt.xlabel("Tiltboard Test Date (2025)")
plt.ylabel("Measured Height (cm)")
plt.title("Tiltboard Test Heights and Fracture Heights")
plt.show()