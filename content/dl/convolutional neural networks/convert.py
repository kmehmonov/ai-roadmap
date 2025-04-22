from moviepy import VideoFileClip

clip = VideoFileClip("images/image_convolution_animated.mp4")
clip.write_gif("images/image_convolution_animated.gif")
