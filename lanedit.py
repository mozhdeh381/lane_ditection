import cv2
import numpy as np

# image = cv2.imread('2.jpg')

# video = cv2.VideoCapture(0) 
   
image = cv2.VideoCapture('./project_video.mp4')   
# We need to check if camera 
# is opened previously or not 
if (image.isOpened() == False):  
    print("Error reading video file") 
  
# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(image.get(3)) 
frame_height = int(image.get(4)) 
   
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 


  
# Check if camera opened successfully 
if (image.isOpened()== False): 
    print("Error opening video file") 
  
# Read until video is completed 
while(image.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = image.read() 
    if ret == True: 

        threshold_max=(900, 255)
        # gray = cv2.cvtcolor(frame,cv2.COLOR_RGB2GRAY)
        hls  = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        s = hls[:,:,2] 
        sobels = cv2.Sobel(s, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobels)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        
        sxbinary[(s >= threshold_max[0]) & (s <= threshold_max[1])] = 1

        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, sxbinary)) * 255

        # Combine the two binary thresholds 
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(sxbinary == 1) | (sxbinary == 1)] = 1
        
       
        img_size = (1100, 1100)

        src = np.float32([(300, 700),
                  (500, 510),
                  (1010, 710),
                  (800, 500)])

        dst = np.float32([(390, 700),
                  (400, 200),
                  (1000, 600),
                  (1000, 200)])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        binary_warped = cv2.warpPerspective(frame, M, img_size, flags=cv2.INTER_LINEAR)
        

        # histogram 

        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # sliding
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50


        # x ,y az binary bedast miad
        window_height = np.int32(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(sxbinary,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2) 
            cv2.rectangle(sxbinary,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)  

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty


        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        unwarp = np.zeros_like(out_img)
        curve = np.column_stack((left_fitx.astype(np.int32), ploty.astype(np.int32)))
        cv2.polylines(unwarp, [curve], False, (0, 255, 255), 6)
        curve = np.column_stack((right_fitx.astype(np.int32), ploty.astype(np.int32)))
        cv2.polylines(unwarp, [curve], False, (0, 255, 255), 6)

        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient

        leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                            for y in ploty])
        rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                            for y in ploty])

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit a second order polynomial to pixel positions in each fake lane line
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        y_eval = np.max(ploty)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


        
        cv2.imshow('sliding', binary_warped)
     
    # Press Q on keyboard to exit 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
 
  

image.release() 
  

cv2.destroyAllWindows()

# draw

