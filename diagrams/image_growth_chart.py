import matplotlib.pyplot as plt
import numpy as np

# Create a simple growth chart
years = np.array([2010, 2015, 2020, 2025])
images_billions = np.array([0.5, 2, 10, 25])  # Estimated numbers

plt.figure(figsize=(8, 5))
plt.plot(years, images_billions, 'bo-', linewidth=3, markersize=8)
plt.fill_between(years, images_billions, alpha=0.3)
plt.title('Digital Images: Exponential Growth', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Images (Billions)')
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.ylim(0, 30)

# Add some annotations
plt.text(2020, 12, 'Social Media\nMobile Devices\nCloud Storage', 
         fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

plt.tight_layout()
plt.savefig('image_growth_chart.png', dpi=150, bbox_inches='tight')
plt.show()