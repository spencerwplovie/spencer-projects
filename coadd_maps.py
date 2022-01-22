#!/usr/bin/env python
# coding: utf-8

## Python 3.8.3 ##

## All imports ##

#General
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#Read in and view FITS file
import os
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS #World Coordinate System
import matplotlib.gridspec as gridspec #Grid layout to place subplots within a figure

#Canny edge detection
from skimage.feature import canny

#Linear Hough Transform
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage import data

#Animation/GIF
from matplotlib import animation

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#Circular Hough Transform
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

#Probabilistic Hough Transform
from skimage.transform import probabilistic_hough_line

#For Gaussian fit
from astropy.modeling import models
from specutils.fitting import fit_lines


## Reading in + viewing the .fits file ##

# file = 'OPH_CORE_20160115_00084_850_EA3_cal.FITS'

file = 'OPH_CORE_850_EA3_cal_smooth_coadd.FITS'

hdulist = fits.open(file) #List of Header Data Units


hdu = hdulist[0]
image = hdulist['PRIMARY',1].data[0]
image1 = np.copy(image) #Creating a separate 'image' variable to overlay in a plot later
wcs = WCS(hdulist[0].header).celestial #World Coordinate System info

#Plotting the input fits file
# #Adding axes to the figure by using gridspec structure
# gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,0.05])
# gs.update(left=0.16, right=0.85, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
# plt.rcParams['axes.titlesize'] = 25
# axs = plt.subplot(gs[0,0], projection=wcs, facecolor='w')
#
#
# #Plot info
# lon = axs.coords[0]
# lat = axs.coords[1]
#
# lon.set_axislabel('Right Ascension', minpad=0.75, fontsize=20)
# lat.set_axislabel('Declination', minpad=-0.3, fontsize=20)
#
# lon.set_ticklabel(size=15, exclude_overlapping=True)
# lat.set_ticklabel(size=15, exclude_overlapping=True)
#
# lon.set_major_formatter('hh:mm:ss')
# lon.set_separator(('h','m','s'))
# lat.set_major_formatter('hh:mm:ss')
#
# lon.set_ticks(spacing=60*u.arcsec)
# lat.set_ticks(spacing=60*u.arcsec)
#
# img = axs.imshow(image, cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gist_yarg colormap scheme
#
#
# #Colorbar
# cb_axs = plt.subplot(gs[0,1]) #Colorbar axes
# cb = plt.colorbar(cax=cb_axs, mappable=img, orientation='vertical', ticklocation='right') #Colorbar
# cb.ax.tick_params(labelsize=20)
# cb.set_label('Intensity [mJy/beam]', fontsize=18, labelpad=20)
#
# axs.set_title('OPH_CORE')
#
# # plt.savefig('OPH_CORE_850_EA3_cal_smooth_coadd.png')
#
# plt.show()


# ## Using numpy roll to shift pixels ##
# ## for x- and y-axis ##
#
# h = 1 #1=3"
#
# #axis = 1 = x-direction
# xrollp = np.roll(image,h,axis=1) #x-direction roll, positive
# xrolln = np.roll(image,-h,axis=1) #x-direction roll, negative
# xdiff = xrollp - xrolln
#
# yrollp = np.roll(image,h,axis=0) #x-direction roll, positive
# yrolln = np.roll(image,-h,axis=0) #x-direction roll, negative
# ydiff = yrollp - yrolln



# ## Plotting df/dx ##
# ## Saving to a new FITS file ##
#
# #Adding axes to the figure by using gridspec structure
# gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,0.05])
# gs.update(left=0.16, right=0.85, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
# plt.rcParams['axes.titlesize'] = 25
# axs = plt.subplot(gs[0,0], projection=wcs, facecolor='w')
#
#
# #Plot info
# lon = axs.coords[0]
# lat = axs.coords[1]
#
# lon.set_axislabel('Right Ascension', minpad=0.75, fontsize=20)
# lat.set_axislabel('Declination', minpad=-0.3, fontsize=20)
#
# lon.set_ticklabel(size=15, exclude_overlapping=True)
# lat.set_ticklabel(size=15, exclude_overlapping=True)
#
# lon.set_major_formatter('hh:mm:ss')
# lon.set_separator(('h','m','s'))
# lat.set_major_formatter('hh:mm:ss')
#
# lon.set_ticks(spacing=60*u.arcsec)
# lat.set_ticks(spacing=60*u.arcsec)
#
#
# ximg = axs.imshow(xdiff, cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gray colormap scheme
#
# # #Smoothed
# # ximg_sm = axs.imshow(xdiff/(2*h), cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gray colormap scheme
#
#
# #Colorbar
# cb_axs = plt.subplot(gs[0,1]) #Colorbar axes
# cb = plt.colorbar(cax=cb_axs, mappable=ximg, orientation='vertical', ticklocation='right') #Colorbar
#
# cb.ax.tick_params(labelsize=20)
# cb.set_label('Intensity [mJy/beam]', fontsize=18, labelpad=20)
#
# axs.set_title('Shift +x')
#
# # hdu1 = fits.PrimaryHDU(xdiff) #Defining a new PrimaryHDU object to write to
# # xdiff_filename = 'OPH_CORE_20160115_00084_850_EA3_cal_xdiff.FITS'
# # hdu1.writeto(xdiff_filename) #Writing the df/dx info to hdu1
#
# plt.show()



# ## Plotting df/dy ##
# ## Saving to a new FITS file ##
#
# #Adding axes to the figure by using gridspec structure
# gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,0.05])
# gs.update(left=0.16, right=0.85, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
# plt.rcParams['axes.titlesize'] = 25
# axs = plt.subplot(gs[0,0], projection=wcs, facecolor='w')
#
#
# #Plot info
# lon = axs.coords[0]
# lat = axs.coords[1]
#
# lon.set_axislabel('Right Ascension', minpad=0.75, fontsize=20)
# lat.set_axislabel('Declination', minpad=-0.3, fontsize=20)
#
# lon.set_ticklabel(size=15, exclude_overlapping=True)
# lat.set_ticklabel(size=15, exclude_overlapping=True)
#
# lon.set_major_formatter('hh:mm:ss')
# lon.set_separator(('h','m','s'))
# lat.set_major_formatter('hh:mm:ss')
#
# lon.set_ticks(spacing=60*u.arcsec)
# lat.set_ticks(spacing=60*u.arcsec)
#
#
# yimg = axs.imshow(ydiff, cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gray colormap scheme
#
# # #Smoothed
# # yimg_sm = axs.imshow(ydiff/(2*h), cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gray colormap scheme
#
#
# #Colorbar
# cb_axs = plt.subplot(gs[0,1]) #Colorbar axes
# cb = plt.colorbar(cax=cb_axs, mappable=yimg, orientation='vertical', ticklocation='right') #Colorbar
#
# cb.ax.tick_params(labelsize=20)
# cb.set_label('Intensity [mJy/beam]', fontsize=18, labelpad=20)
#
# axs.set_title('Shift +y')
#
# # hdu2 = fits.PrimaryHDU(ydiff) #Defining a new PrimaryHDU object to write to
# # ydiff_filename = 'OPH_CORE_20160115_00084_850_EA3_cal_ydiff.FITS'
# # hdu1.writeto(ydiff_filename) #Writing the df/dx info to hdu1
#
# plt.show()



# ## Calculating df/dx, dy/dx, mag/angle of gradient, divergence ##
#
# dfdx = xdiff/(2*h)
# dfdy = ydiff/(2*h)
#
# #Gradient
# grad_mag = np.sqrt((dfdx)**2 + (dfdy)**2)
# grad_theta = np.arctan((dfdy)/(dfdx))
#
# #Divergence
# divergx = (xrollp - 2*image + xrolln)/h
# divergy = (yrollp - 2*image + yrolln)/h
# diverg = divergx + divergy



## Plotting grad_mag ##
## Different colorbar bounds!! ##

# #Adding axes to the figure by using gridspec structure
# gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,0.05])
# gs.update(left=0.16, right=0.85, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
# plt.rcParams['axes.titlesize'] = 25
# axs = plt.subplot(gs[0,0], projection=wcs, facecolor='w')


# #Plot info
# lon = axs.coords[0]
# lat = axs.coords[1]
#
# lon.set_axislabel('Right Ascension', minpad=0.75, fontsize=20)
# lat.set_axislabel('Declination', minpad=-0.3, fontsize=20)
#
# lon.set_ticklabel(size=15, exclude_overlapping=True)
# lat.set_ticklabel(size=15, exclude_overlapping=True)
#
# lon.set_major_formatter('hh:mm:ss')
# lon.set_separator(('h','m','s'))
# lat.set_major_formatter('hh:mm:ss')
#
# lon.set_ticks(spacing=60*u.arcsec)
# lat.set_ticks(spacing=60*u.arcsec)
#
#
# gradimg = axs.imshow(grad_mag, cmap=plt.cm.gray, aspect='equal', vmin=0, vmax=0.1) #gray colormap scheme
#
# #Colorbar
# cb_axs = plt.subplot(gs[0,1]) #Colorbar axes
# cb = plt.colorbar(cax=cb_axs, mappable=gradimg, orientation='vertical', ticklocation='right') #Colorbar
#
# cb.ax.tick_params(labelsize=20)
# cb.set_label('Intensity [mJy/beam]', fontsize=18, labelpad=20)
#
# axs.set_title('Gradient Magnitude')
#
# # hdu3 = fits.PrimaryHDU(grad_mag) #Defining a new PrimaryHDU object to write to
# # grad_filename = 'OPH_CORE_20160115_00084_850_EA3_cal_gradient.FITS'
# # hdu3.writeto(grad_filename) #Writing the df/dx info to hdu1
#
# plt.show()



# ## Plotting the divergence ##
# ## Saving to a new FITS file ##
#
# #Adding axes to the figure by using gridspec structure
# gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,0.05])
# gs.update(left=0.16, right=0.85, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
# plt.rcParams['axes.titlesize'] = 25
# axs = plt.subplot(gs[0,0], projection=wcs, facecolor='w')
#
#
# #Plot info
# lon = axs.coords[0]
# lat = axs.coords[1]
#
# lon.set_axislabel('Right Ascension', minpad=0.75, fontsize=20)
# lat.set_axislabel('Declination', minpad=-0.3, fontsize=20)
#
# lon.set_ticklabel(size=15, exclude_overlapping=True)
# lat.set_ticklabel(size=15, exclude_overlapping=True)
#
# lon.set_major_formatter('hh:mm:ss')
# lon.set_separator(('h','m','s'))
# lat.set_major_formatter('hh:mm:ss')
#
# lon.set_ticks(spacing=60*u.arcsec)
# lat.set_ticks(spacing=60*u.arcsec)
#
#
# divimg = axs.imshow(diverg, cmap=plt.cm.gray, aspect='equal', vmin=-0.1, vmax=0.1) #gray colormap scheme
#
# #Colorbar
# cb_axs = plt.subplot(gs[0,1]) #Colorbar axes
# cb = plt.colorbar(cax=cb_axs, mappable=divimg, orientation='vertical', ticklocation='right') #Colorbar
#
# cb.ax.tick_params(labelsize=20)
# cb.set_label('Intensity [mJy/beam]', fontsize=18, labelpad=20)
#
# axs.set_title('Divergence')
#
# # hdu4 = fits.PrimaryHDU(diverg) #Defining a new PrimaryHDU object to write to
# # diverg_filename = 'OPH_CORE_20160115_00084_850_EA3_cal_divergence.FITS'
# # hdu4.writeto(diverg_filename) #Writing the df/dx info to hdu1
#
# plt.show()



# ## Creating a cropped version of image ##

# trim = 200
# image1 = image[trim:-trim,trim:-trim]
# image1[image1 < 0.01] = 0 #Sets a threshold value for pixels, 0.05 = 50mJy
# # image1 = image1[::-1] #Vertically flips image1

##


## Changing NaN to zeroes, sets threshold value ##
## Setting vmin, vmax, and threshold values for following HT plots ##

image[np.isnan(image)==True] = 0 #Instead of cropping the image
image1[np.isnan(image1)==True] = 0
image[image < 0.01] = 0 #Pixel threshold, 0.01 = 10mJy. Replaced with line below.

# image[image < 0.006] = 0 #Pixel threshold, 0.006 = 6mJy
# image1[image1 < 0.006] = 0
image1[image1 != 0] = 1 #Image disk

vmin = -0.1
vmax = 0.1

th = 250 #Threshold value used to find HT peaks

# print("Maximum pixel value: %.3f Jy" % np.nanmax(image)) #NOTE: Jy, not in Jy/beam
#nanmax excludes any nan values encountered!
#Should be approx. 4 Jy?



## Creating the linear Hough transform ##
## Plotting the original image, HT, and detected lines ##

#Precision of 0.5 degrees
tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)
h1, theta1, d1 = hough_line(image1, theta=tested_angles)
#h = accumulator space

# print(np.shape(h))

#Generating the plot format

# # Trying to change the dimensions for each plot separately
# widths = [12, 20, 12]
# heights = [12]
# gs_kw = dict(width_ratios=widths, height_ratios=heights)
# fig, axs = plt.subplots(1,3, gridspec_kw=gs_kw)

fig, axs = plt.subplots(1,3, figsize=(15,6))
ax = axs.ravel() #Flatten the array

ax[0].imshow(image, origin='lower', cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax[0].set_title('Input Image')
ax[0].set_axis_off() #No axes labels


#Creating the bounds
angle_step = 0.5*np.diff(theta).mean() #dtheta
d_step =0.5*np.diff(d).mean() #dx
#Creating the (left, right, top, bottom) bounds for the image to be just outside the d and theta arrays
bounds = [np.rad2deg(theta[0] - angle_step), #Subtracts dtheta from first index
          np.rad2deg(theta[-1] + angle_step), #Adds dtheta to last index
          d[-1] + d_step, d[0] - d_step] #Same for d

#Plotting the HT
# ax[1].imshow(np.log(1+h), extent=bounds, cmap=plt.cm.gray, aspect='auto')
ax[1].imshow(h, vmin=th, vmax=th+100, extent=bounds, cmap=plt.cm.gray, aspect='auto')
# ax[1].imshow(np.log(1+h), cmap=plt.cm.gray, aspect='auto') #aspect=float is height:width ratio value
ax[1].set_title('Hough Transform')
ax[1].set_xlabel('Angles (°)')
ax[1].set_ylabel('Distance (pix)')
# ax[1].axis('Image') #Sets axes to fit original data shape, don't want this!

ax[2].imshow(image, origin='lower', cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
# ax[2].set_ylim((image.shape[0],0))
ax[2].set_axis_off()
ax[2].set_title('Detected Lines')
trim = 0

theta_list = []
#Plotting the detected lines
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=th)): #_ is a filler, we just want angle, dist
    (x0,y0) = dist*np.array([np.cos(angle), np.sin(angle)])

    theta_list.append(angle) #Appending angles found in image to a separate list

    # if (angle > -np.pi/3) and (angle < -np.pi/12):
    ax[2].axline((x0,y0), slope=np.tan(angle+np.pi/2), alpha=0.5, color='r')
    ax[2].set_ylim((0,len(image)-trim*2))
    ax[2].set_xlim((0,len(image)-trim*2))


plt.tight_layout()

# plt.savefig('Input_HT_Detected.png')
plt.show()


## Histogram of 'h' values from HT ##

# plt.hist(h.ravel(), range=(th,400))



## Creating a Histogram Plot of the HT Angles ##

# base = abs(theta[0]) #Not a good distribution, maybe fix bins_theta?
# base = 1.2
# bins_theta = [(base**(i)-2.5) for i in range(10)] #Exponential bins
bins_theta = np.linspace(-np.pi/2,np.pi/2,19) #Linear bins
# (n,bins,patches) = plt.hist(theta_list, bins=bins_theta)
(n,bins,patches) = plt.hist(theta_list, bins_theta)

# plt.xlim((-1.5,1.5))
# plt.xlim((-2,2))
plt.title('Frequency of Angles in Hough Transform', fontsize=16)
plt.xlabel('Angles (radians)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

print("Bins for angles (in degrees):")
print(np.multiply(bins_theta,(180/(np.pi))))
# print()
# print("Bins for angles:")
# print(bins_theta)
print()
print("Frequency of values in bins_theta:")
print(n)



## Finding the range where the maximum number of lines are detected ##

theta_max_ind = np.argmax(n) #Finds the index of the highest-frequency bin
print("Maximum number of lines detected at angles between %.2f° and %.2f° "  % (bins_theta[theta_max_ind]*(180/(np.pi)),bins_theta[theta_max_ind+1]*(180/(np.pi))) )
# print("Maximum number of lines detected at angles between %.2f and %.2f radians "  % (bins_theta[theta_max_ind],bins_theta[theta_max_ind+1]))



# ## Plotting just the detected lines image ##
#
# plt.figure(figsize=(7,7))
# plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
# plt.ylim((image.shape[0],0))
# # ax[2].set_axis_off()
# plt.title('Detected Lines')
#
# #Plotting the detected lines
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=th)): #_ is a filler, we just want angle, dist
#     (x0,y0) = dist*np.array([np.cos(angle), np.sin(angle)])
#     plt.axline((x0,y0), slope=np.tan(angle+np.pi/2), alpha=0.5, color='r')
#     plt.ylim((0,len(image)-trim*2))
#     plt.xlim((0,len(image)-trim*2))
#
# plt.show()

# print(h[500:510,0])

# plt.plot(h[:,0])
# plt.show()

# x_sum = np.sum(h[:,60:79],axis=1)/20 #Grabs all from accumulator for 10-degree increments


num_sums = 18
# x_sum = np.zeros(num_sums)
x_sum = [[] for i in range(num_sums)]
x_sum1 = [[] for i in range(num_sums)]
for i in range(num_sums):
    x_sum[i] = np.sum(h[:, i*20:i*20+19], axis=1)/20  # Grabs all from accumulator for 10-degree increments
    x_sum1[i] = np.sum(h1[:, i*20:i*20+19], axis=1)/20
#Normalized by 20 = 10 degrees, since each integer is half a degree (see 'tested_angles')
#x_sum takes the sum of all the distances (and normalizes by 10 degrees) so the distances are all the same for that 10 degree increment

# for i in range(num_sums):
#     #Plots individual frames
#     plt.plot(x_sum[i])
#     plt.title("Accumulator Space")
#     plt.xlim((0,2500))
#     plt.show()


# ## Creating a 3x6 plot for all frames ##
#
# fig, axs = plt.subplots(6,3)
#
# index = []
# for i in range(6):
#     for j in range(3):
#         index.append((i,j)) #For plotting in [row,col]
#
# for i in range(num_sums):
#     axs[index[i]].plot(x_sum[i])
#
# plt.xlim((0,2500))
# plt.show()



## Saving all frames individually ##
#To later combine into a PDF

# for i in range(num_sums):
#     plt.plot(x_sum1[i],color='k',linestyle='dashed',alpha=0.6) #Maximum possible value for each 10-degree interval (envelope)
#     plt.plot(x_sum[i],color='k') #Plots overtop of the 'image disk' (range of possible values)
#     plt.plot(2000*np.divide(x_sum[i],x_sum1[i]),color='r')
#     #Dividing solid by dashed tells you what fraction of that 'chord' (angle) was associated with a line
#     #Multiply by 2000 so it's easier to see
#     #Looking for evidence of scales - tells us things are aligned along that axis
#     plt.ylim((0,850))
#     # plt.ylim((0,300))
#     plt.text(1800,700,'θ=['+str(i*10)+','+str(i*10+10)+']°')
#     plt.title('h values (10° increments)')
#     # plt.savefig('picture-%d.png' % i)
#     # plt.savefig('picture1-%d.png' % i) #Uncomment this when plotting image disk
#     plt.savefig('picture1_{}'.format(i))
#     plt.show()
#Don't need to run every time
#Plotting x_sum allows us to see what h looks like at different angles



#Terminal command to create the video:
# ffmpeg -r 5 -i picture-%01d.png -c:v libx264 -r 20 -pix_fmt yuv420p hsum_anim.mp4
#5 = 1/time allowed for each frame
#20 = fps

#Terminal command to create the video INCLUDING image disk
# ffmpeg -r 5 -i picture1-%01d.png -c:v libx264 -r 20 -pix_fmt yuv420p hsum_anim1.mp4




#To-do:
#Try FAS/Power Spectrum for both black and red lines (FFT?)
#Gaussian fit for red and black lines - get sigma, amplitude, and mean
#Try correlation thing?
#Reminder: Doug mostly unavailable for next 2-3 weeks
#Post whatever you finish on Slack

#At 37:00 in last Zoom discussion - created plots, discussing distance scales (x-axis)


def pdf(x_sum):
    mu = np.mean(x_sum)
    sigma = np.std(x_sum)

    f = (1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-0.5*((x_sum-mu)/(sigma))**2))
    return f

gauss = pdf(x_sum[6])


# gi = models.Gaussian1D(amplitude=200, mean=np.mean(x_sum[6]), stddev=1)
# gfit = fit_lines(x_sum[6], gi)
# yfit = g_fit(x_sum[6])

test = np.linspace(0, len(x_sum[6]), len(x_sum[6]))

plt.plot(test, gauss, label='AHHH')
# plt.plot(x_sum[6], label='Original')
# plt.plot(gauss, label='Gaussian')
plt.title('Single fit peak')
plt.legend()
plt.show()



# # Fit the spectrum and calculate the fitted flux values (``y_fit``)
# g_init = models.Gaussian1D(amplitude=3.*u.Jy, mean=6.1*u.um, stddev=1.*u.um)
# g_fit = fit_lines(spectrum, g_init)
# y_fit = g_fit(x*u.um)
#
# # Plot the original spectrum and the fitted.
# plt.plot(x, y, label="Original spectrum")
# plt.plot(x, y_fit, label="Fit result")
# plt.title('Single fit peak')
# plt.grid(True)
# plt.legend()


## Creating the Fourier Amplitude Spectrum for one of the frames ##






##



##For correlation coefficients in future, we can look at this in detail together.
# #I have to see what shape the values in the y axis have. Let’s assume your y axis is a numpy array with a shape of (800, 801),
# #and the pixels of your raw image have a (800,801) shape. In that case you can find the correlation coefficient between the two using:
#Data = np.reshape(image, (1,-1))[0]
##Same thing for your y array (you need to flatten the nparrays to be able to find the correlation coefficients):
#Data2 = np.reshape(y-axis, (1,-1))[0]
#from scipy.stats.stats import pearsonr
#Correlationcoefficient = pearsonr(ExtnctData[indNAN], velData[indNAN])[0]This gives you a value to see whether you see a positive or negative (anticorrelation) between the peaks of your y-axis and the peaks in your image.

##



# -make animations (Mehrnoosh code, PDF style)
# -update so you can do HT AND for disc one
# -take any one of the panels and see if you can find a program that gives you a power spectrum and plot it (it will look weird, FYI)




# ## Creating a gif from the 18 frames ##
# 
# fig, axs = plt.subplots()
# 
# def update_plot(i):
#   for i in range(num_sums):
#     axs.plot(x_sum[i])
#     axs.set_xlim((0,2500))
# 
# anim = animation.FuncAnimation(fig, update_plot, frames = num_sums)
# anim.save('test.gif')



# ## Creating a gif from the 18 frames - method 2 ##
#
# duration = 2 #Duration of video
# fig, axs = plt.subplots()
#
# def make_frame(time):
#
#     axs.clear()
#
#     axs.plot(x_sum[i])
#     axs.set_xlim((0,2500))
#
#     return mplfig_to_npimage(fig) #Returns numpy image
#
# anim = VideoClip(make_frame, duration=duration)
#
# animation.ipython_display(fps=10, loop=True, autoplay=True)





# In[ ]:


# ## Creating the linear probabilistic Hough transform ##
# ## Plotting the original image, Canny edge detection, and HT ##

# #Fiddle with these parameters to change 2nd and 3rd image
# low = -0.1
# high = 0.1

# edges = canny(image[::-1], sigma=3, low_threshold=low, high_threshold=high)
# lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
#                                  line_gap=3)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(image[::-1], cmap=plt.cm.gray, vmin=-0.1, vmax=0.1)
# ax[0].set_title('Input image')

# ax[1].imshow(edges, cmap=plt.cm.gray)
# ax[1].set_title('Canny edges')

# ax[2].imshow(edges*0)
# for line in lines:
#     p0, p1 = line
#     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax[2].set_xlim((0, image.shape[1]))
# ax[2].set_ylim((image.shape[0], 0))
# ax[2].set_title('Probabilistic Hough')

# for a in ax:
#     a.set_axis_off()

# plt.tight_layout()
# plt.show()


# In[ ]:


# ## Creating the circular Hough transform ##

# #Detecting the edges
# edges = canny(image[::-1], sigma=3, low_threshold=low, high_threshold=high)

# #Detect two radii
# hough_radii = np.arange(20,35,2)
# hough_res = hough_circle(edges,hough_radii)

# #Select the most prominent 3 circles
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

# #Drawing the circles
# fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(10,4))
# # image_grey = color.gray2rgb(image)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
# #     image[circy, circx] = (220,20,20)
#     image[circy, circx] = (220)
    
# axs.imshow(image[::-1], cmap=plt.cm.gray, vmin=-0.1, vmax=0.1)
# plt.show()
