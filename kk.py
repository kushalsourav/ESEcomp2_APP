# import plotly.graph_objects as go
# import pandas as pd

# # Data
# data = pd.read_csv("Models/02_captcha_to_text/202401211802/val.csv")

# # Convert data to numeric (if necessary)
# numeric_data = data[1]

# # Plot 1: Histogram
# fig1 = go.Figure(data=[go.Histogram(x=numeric_data)])
# fig1.update_layout(title='Histogram of Data', xaxis_title='Value', yaxis_title='Frequency')

# # Plot 2: Box Plot
# fig2 = go.Figure(data=go.Box(y=numeric_data, boxpoints='all', jitter=0.3, pointpos=-1.8))
# fig2.update_layout(title='Box Plot of Data', yaxis_title='Value')

# # Plot 3: Scatter Plot
# fig3 = go.Figure(data=go.Scatter(x=list(range(len(numeric_data))), y=numeric_data, mode='markers'))
# fig3.update_layout(title='Scatter Plot of Data', xaxis_title='Index', yaxis_title='Value')

# # Plot 4: Violin Plot
# fig4 = go.Figure(data=go.Violin(y=numeric_data, box_visible=True, line_color='black', meanline_visible=True,
#                                  fillcolor='lightseagreen', opacity=0.6))
# fig4.update_layout(title='Violin Plot of Data', yaxis_title='Value')

# # Plot 5: Pie Chart
# value_counts = {val: data.count(val) for val in set(data)}
# labels = list(value_counts.keys())
# values = list(value_counts.values())
# fig5 = go.Figure(data=[go.Pie(labels=labels, values=values)])
# fig5.update_layout(title='Pie Chart of Data')

# # Show plots
# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()



# import matplotlib.pyplot as plt

# # List of strings
# data = [
#     '1', '7cd4bd', '7fc0d9', 'a35607', '7381c8', '1e76de', '44d23f', '656474', '9f7ade', '51bc1f',
#     '4f402d', '602bf5', 'e82ef0', 'c037a5', 'deb035', '0cc5e7', 'c63d7c', 'ff11ec', '6548', '3a6752',
#     '80527b', '834d6f', '58ec67', 'b46523', '9f97e5', '01f808', '9.27E+36', '85566f', '3304fd', 'aaa5d9',
#     'aced78', 'ea3b0a', '91843b', '35a9b2', 'f5c0fd', '655194', 'c1a708', '34ee0a', 'ea8588', '034dda',
#     'bcb285', '9bfdb1', '427a74', '1751b1', 'a51c08', 'edd67e', 'f40cc7', '1a028c', 'fa5740', '5cd931',
#     'fb8355', '510839', '10bba6', '204481', '6f4c61', '71a6ff', '14c2fa', '3c6b6a', '4e2279', '664ee6',
#     'e99fc9', 'b109c1', 'f1563f', '1c216e', 'a204c0', '1a7408', 'fc0ec9', '8ce1e5', '281fb0', '24f55b',
#     '3c7349', '2eaef3', '496c85', '0afd59', 'c6ba19', 'aa5689', 'e89c20', '68083', '8e3fbd', 'ead25b',
#     'be589f', 'f4e31c'
# ]


# lengths = [len(s) for s in data]


# plt.figure(figsize=(10, 6))
# plt.hist(lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
# plt.title('Distribution of String Lengths')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()



# alpha_counts = [sum(c.isalpha() for c in s) for s in data]


# plt.figure(figsize=(12, 6))
# plt.bar(range(len(data)), alpha_counts, color='skyblue', edgecolor='black', alpha=0.7)
# plt.title('Number of Alphabetic Characters in Each String')
# plt.xlabel('String')
# plt.ylabel('Count of Alphabetic Characters')
# plt.grid(True)
# plt.show()


# # Count numeric characters in each string
# numeric_counts = [sum(c.isdigit() for c in s) for s in data]

# # Plotting bar plot
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(data)), numeric_counts, color='lightgreen', edgecolor='black', alpha=0.7)
# plt.title('Number of Numeric Characters in Each String')
# plt.xlabel('String')
# plt.ylabel('Count of Numeric Characters')
# plt.grid(True)
# plt.show()

# import random
# import string
# from PIL import Image, ImageDraw, ImageFont

# def generate_random_text(length=6):
#     characters = string.ascii_letters + string.digits
#     return ''.join(random.choice(characters) for _ in range(length))


# from PIL import Image, ImageDraw, ImageFont

# def generate_image_with_text(text, crop_size=(418, 468, 580, 517), image_size=(1000, 1000), background_color=(255, 255, 255)):
#     # Create a new image with the specified size and background color
#     image = Image.new('RGB', image_size, color=background_color)
    
#     # Define the cropping region
#     left, top, right, bottom = crop_size
    
#     # Create a cropped image
#     cropped_image = image.crop((left, top, right, bottom))

#     # Get a drawing context
#     draw = ImageDraw.Draw(cropped_image)

#     font_path = "C:/Windows/Fonts/Arial.ttf"  # Example font path, change it to match your system font file
#     font = ImageFont.truetype(font_path, 30)
   
#     draw.text((10,10),text, fill=(0, 0, 0), font=font)

#     # Show the cropped image with text
#     cropped_image.show()

# if __name__ == "__main__":
#     captcha_text = generate_random_text()
#     print("Captcha Text:", captcha_text)
#     generate_image_with_text(captcha_text)


import random
import string
from PIL import Image, ImageDraw, ImageFont

def generate_random_text(length=6):
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def generate_image_with_text(text, crop_size=(418, 468, 580, 517), image_size=(1000, 1000), background_color=(255, 255, 255)):
  
    image = Image.new('RGB', image_size, color=background_color)

    left, top, right, bottom = crop_size

    cropped_image = image.crop((left, top, right, bottom))

    draw = ImageDraw.Draw(cropped_image)

    font_path = "C:/Windows/Fonts/Arial.ttf"  
    font = ImageFont.truetype(font_path, 30)

    x_position = 10
    for char in text:
    
        text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text((x_position, 10), char, fill=text_color, font=font)
       
        x_position += 25  

    cropped_image.show()

if __name__ == "__main__":
    captcha_text = generate_random_text()
    print("Captcha Text:", captcha_text)
    generate_image_with_text(captcha_text)
