import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read figure and classify according to Plumes, Escape cells, and Necrotic core 
def ImageOpenCV(file,plot=True):
    img = cv2.imread(file)
    blur = cv2.blur(img,(5,5))
    
    # Cutting image
    img = img[100:-70, 5:-5]
    #img = img[250:-150, 150:-150]
    
    # Red filter
    channelred = img[:,:,0]
    _,thresh_val1 = cv2.threshold(channelred,15 ,255,cv2.THRESH_BINARY)
    thresh_val1 = cv2.bitwise_not(thresh_val1)
    # Green filter
    channelgreen = img[:,:,1]
    _,thresh_val2 = cv2.threshold(channelgreen,70 ,255,cv2.THRESH_BINARY)
    thresh_val2 = cv2.bitwise_not(thresh_val2)
    # Blue filter
    channelblue = img[:,:,2]
    _,thresh_val3 = cv2.threshold(channelblue,40 ,255,cv2.THRESH_BINARY)
    thresh_val3 = cv2.bitwise_not(thresh_val3)  
    # Combine
    mask = cv2.bitwise_and(thresh_val1,thresh_val2)
    mask2 = cv2.bitwise_and(thresh_val1,~mask)
    
    # Setting the elements
    element = cv2.getStructuringElement(shape=cv2.MORPH_OPEN, ksize=(5, 5))
    element2 = cv2.getStructuringElement(shape=cv2.MORPH_OPEN, ksize=(8, 8))

    morph_img = thresh_val1.copy()
    cv2.morphologyEx(src=thresh_val1, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)

    morph_img2 = mask2.copy()
    cv2.morphologyEx(src=mask2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img2)

    morph_img3 = thresh_val3.copy()
    cv2.morphologyEx(src=thresh_val3, op=cv2.MORPH_CLOSE, kernel=element2, dst=morph_img3)
    
    # Obtaining contours from filters
    contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours2,_ = cv2.findContours(morph_img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours3,_ = cv2.findContours(morph_img3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Sorting areas to each countour
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    areas2 = [cv2.contourArea(c) for c in contours2]
    sorted_areas2 = np.sort(areas2)
    areas3 = [cv2.contourArea(c) for c in contours3]    
    
    #Ellipses
    img1 = Convert_BW_to_RGB(morph_img2)
    img2 = Convert_BW_to_RGB(morph_img+morph_img3)
    if (len(sorted_areas) > 0): 
        cntA=contours[areas.index(sorted_areas[-1])] #the biggest contour
        ellipseA = cv2.fitEllipse(cntA)
        cv2.ellipse(img2,ellipseA,(0,0,200),4)    
    if (len(sorted_areas2) > 0): 
        cntA2=contours2[areas2.index(sorted_areas2[-1])] #the biggest contour
        epsilon = 0.00001 * cv2.arcLength(cntA2, True)
        approx = cv2.approxPolyDP(cntA2, epsilon, True)
        cv2.drawContours(img1, [approx], -1, (0, 255, 0), 4)
        ellipseA2 = cv2.fitEllipse(cntA2)
        cv2.ellipse(img1,ellipseA2,(0,120,200),4) 

    #Necrotic cells
    img3 = morph_img3 - (morph_img)
    necrotic = False
    if( (np.sum(areas3)/np.sum(areas)) > 0.06):
        necrotic = True   
    
    #Scapping cells
    scape = False  
    morph_img4 = morph_img2.copy()
    (h,k),(ma,MA),angle = ellipseA
    ma = ma + 0.05*ma
    MA = MA + 0.05*MA 
    ellipseB = (h,k),(ma,MA),angle 
    cv2.ellipse(morph_img4,ellipseB,(0,0,0),-1)
    contours4,_ = cv2.findContours(morph_img4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas4 = [cv2.contourArea(c) for c in contours4]
    if (np.sum(areas4) > 20.0):
        scape = True
    for ind in range(0,len(contours4)): 
        x = np.mean(contours4[ind][:,0,0])
        y = np.mean(contours4[ind][:,0,1])
        cv2.circle(img2,(int(x),int(y)),4,(0,200,0),2)

    #Plumes test
    plumes = False
    count = 0
    if (len(sorted_areas2) > 0): 
        (h,k),(ma,MA),angle = ellipseA2
        angle = angle*np.pi/180.0
    for i in range( 0,approx.shape[0]):
        (x,y) = approx[i,0,:]
        p = (((x - h)*np.cos(angle) + (y-k)*np.sin(angle))**2 / ((0.5*ma)**2)) +  (((x - h)*np.sin(angle) - (y-k)*np.cos(angle))**2 / ((0.5*MA)**2))
        if (p > 1.0): 
            dist = estimate_distance(x, y, 0.5*MA, 0.5*ma, x0=h, y0=k, angle=0, error=1e-5)
            tol =30.0
            if (dist >  tol):
                plumes = True

    if (plot):
        cv2.imshow("Original", img)
        cv2.imshow("Plumes",img1)
        cv2.imshow("Escape cells",img2)
        cv2.imshow("Necrotic core",img3)
        # Save image
        # cv2.imwrite("Image.jpg", img)
        # cv2.imwrite("Plumes.jpg", img1)
        # cv2.imwrite("Scaping.jpg",img2)
        # cv2.imwrite("Necrotic.jpg", img3)
        cv2.waitKey()

    return plumes, scape, necrotic

def Convert_BW_to_RGB(image):
    New_image = np.zeros((image.shape[0],image.shape[1],3))
    ind = np.argwhere(image==0)
    New_image[ind[:,0],ind[:,1],0] = 0
    New_image[ind[:,0],ind[:,1],1] = 0
    New_image[ind[:,0],ind[:,1],2] = 0
    ind_0 = np.argwhere(image!=0)
    New_image[ind_0[:,0],ind_0[:,1],0] = 255
    New_image[ind_0[:,0],ind_0[:,1],1] = 255
    New_image[ind_0[:,0],ind_0[:,1],2] = 255
    return New_image
    
def Classify_all_simulations():
    Plumes = np.zeros(27)
    EscCell = np.zeros(27)
    NecCore = np.zeros(27)

    Bias = ["00","05","10"]
    Fraction = ["010","050","100"]
    TimePers = ["000","050","999"]

    index1 = []
    index2 = []
    index3 = []
    index4 = []
    index5 = []
    index6 = []
    index7 = []
    index8 = []
    index9 = []

    for n in range(len(Plumes)):
        if (n < 9):
            File = "snapshots/Output_B"+Bias[0]
            index1.append(int(n))
        if ( (n>=9) & (n<18) ):
            File = "snapshots/Output_B"+Bias[1]
            index2.append(int(n))
        if (n>=18):
            File = "snapshots/Output_B"+Bias[2]
            index3.append(int(n))
        if (n%9 < 3):
            File += "_F"+Fraction[0]
            index4.append(int(n))
        if ( (n%9 >= 3) & (n%9 < 6) ):
            File += "_F"+Fraction[1]
            index5.append(int(n))
        if (n%9 >= 6):
            File += "_F"+Fraction[2]
            index6.append(int(n))
        if (n%3 == 0):
            File += "_T"+TimePers[0]
            index7.append(int(n))
        if (n%3 == 1):
            File += "_T"+TimePers[1]
            index8.append(int(n))
        if (n%3 == 2):
            File += "_T"+TimePers[2]
            index9.append(int(n))
        File += ".jpg"
        Plumes[n], EscCell[n], NecCore[n] = ImageOpenCV(File,False)

    #generate data
    a = np.zeros((3, 9))
    a[0,0::3] = Plumes[index4[-3:]]
    a[0,1::3] = EscCell[index4[-3:]]
    a[0,2::3] = NecCore[index4[-3:]]

    a[1,0::3] = Plumes[index4[3:6]]
    a[1,1::3] = EscCell[index4[3:6]]
    a[1,2::3] = NecCore[index4[3:6]]

    a[2,0::3] = Plumes[index4[0:3]]
    a[2,1::3] = EscCell[index4[0:3]]
    a[2,2::3] = NecCore[index4[0:3]]
    discrete_matshow(a)

    b = np.zeros((3, 9))
    b[0,0::3] = Plumes[index5[-3:]]
    b[0,1::3] = EscCell[index5[-3:]]
    b[0,2::3] = NecCore[index5[-3:]]


    b[1,0::3] = Plumes[index5[3:6]]
    b[1,1::3] = EscCell[index5[3:6]]
    b[1,2::3] = NecCore[index5[3:6]]


    b[2,0::3] = Plumes[index5[0:3]]
    b[2,1::3] = EscCell[index5[0:3]]
    b[2,2::3] = NecCore[index5[0:3]]
    discrete_matshow(b)

    c = np.zeros((3, 9))
    c[0,0::3] = Plumes[index6[-3:]]
    c[0,1::3] = EscCell[index6[-3:]]
    c[0,2::3] = NecCore[index6[-3:]]

    c[1,0::3] = Plumes[index6[3:6]]
    c[1,1::3] = EscCell[index6[3:6]]
    c[1,2::3] = NecCore[index6[3:6]]

    c[2,0::3] = Plumes[index6[0:3]]
    c[2,1::3] = EscCell[index6[0:3]]
    c[2,2::3] = NecCore[index6[0:3]]

    discrete_matshow(c)
  
def discrete_matshow(data):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(6.5 ,7))
    #get discrete colormap
    cm = LinearSegmentedColormap.from_list('MyBarColor', [(1.0,0,0),(0,0,1.0)], N=2)
    cmap = plt.get_cmap('RdGy', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = ax.matshow(data,cmap=cm,vmin = np.min(data)-.5, vmax = np.max(data)+.5,aspect='auto')
    #tell the colorbar to tick at integers
    divider = make_axes_locatable(ax)
    bar = divider.append_axes("right", size="5%", pad=0.1)
    cax = plt.colorbar(mat, ticks=[0,1], cax=bar)
    cax.ax.set_yticklabels(['False', 'True'],fontsize=14)
    ax.set_xticks([2.5,5.5])
    ax.set_xticklabels([])
    ax.set_yticks([0.5,1.5])
    ax.set_yticklabels([])
    ax.set_xticks([0.5,1.5,3.5,4.5,6.5,7.5], minor=True)
    plt.text(-39.8, 1.55, 'Plumes', fontsize=12,rotation=90)
    plt.text(-35.4, 1.55, 'Escape', fontsize=12,rotation=90)
    plt.text(-31, 1.55, 'Necrotic', fontsize=12,rotation=90)
    plt.text(-26.6, 1.55, 'Plumes', fontsize=12,rotation=90)
    plt.text(-22.2, 1.55, 'Escape', fontsize=12,rotation=90)
    plt.text(-17.8, 1.55, 'Necrotic', fontsize=12,rotation=90)
    plt.text(-13.4, 1.55, 'Plumes', fontsize=12,rotation=90)
    plt.text(-9.0, 1.55, 'Escape', fontsize=12,rotation=90)
    plt.text(-4.6, 1.55, 'Necrotic', fontsize=12,rotation=90)
    # T_p
    plt.text(-37.6, -0.58, '$T_p = 0h$', fontsize=14)
    plt.text(-24.95, -0.58, '$T_p = 50h$', fontsize=14)
    plt.text(-12.5, -0.58, '$T_p = 130h$', fontsize=14)
    # Bias
    plt.text(-43.4, 1.05, '$b = 1.0$', fontsize=14,rotation=90)
    plt.text(-43.4, 0.4, '$b = 0.5$', fontsize=14,rotation=90)
    plt.text(-43.4, -0.25, '$b = 0.1$', fontsize=14,rotation=90)
    ax.grid(which='minor', linewidth=1)
    ax.grid(color='w', linewidth=6)
    plt.show()
 
def ReplicaAnalysis(folder="replicas"):
    Plumes = np.zeros(20)
    EscCell = np.zeros(20)
    NecCore = np.zeros(20)
  
    for n in range(len(Plumes)):
        File = folder+"/final"+"%02i"%(n+1)+".jpg"
        Plumes[n], EscCell[n], NecCore[n] = ImageOpenCV(File,False)
    
    PlumesT = 100.0*np.sum(Plumes)/len(Plumes)
    PlumesF = 100.0 - PlumesT
    EscapeT = 100.0*np.sum(EscCell)/len(Plumes)
    EscapeF = 100.0 - EscapeT
    NecCorT = 100.0*np.sum(NecCore)/len(Plumes)
    NecCorF = 100.0 - NecCorT
    print(">>> Plumes: " + str(PlumesT) + "% Escape: " + str(EscapeT) + "% NecCore: " + str(NecCorT)+"%")
    
    labels = 'True', 'False'
    colors = ['Blue', 'Red']
    sizesP = [PlumesT, PlumesF]
    sizesE = [EscapeT, EscapeF]
    sizesN = [NecCorT, NecCorF]
    explode = (0, 0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.pie(sizesP, explode=explode, autopct='%1.0f%%', shadow=True, startangle=90,colors=colors,textprops={'fontsize': 16})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2 = fig.add_subplot(1,3,2)
    ax2.pie(sizesE, explode=explode, autopct='%1.0f%%', shadow=True, startangle=90,colors=colors,textprops={'fontsize': 16})
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax3 = fig.add_subplot(1,3,3)
    ax3.pie(sizesN, explode=explode, autopct='%1.0f%%', shadow=True, startangle=90,colors=colors,textprops={'fontsize': 16})
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig.legend(labels, loc="right",prop={'size': 16})
    plt.show()
  
 
from math import sin, cos, atan2, pi, fabs
def ellipe_tan_dot(rx, ry, px, py, theta):
    # Dot product of the equation of the line formed by the point with another point on the ellipse's boundary and the tangent of the ellipse at that point on the boundary.
    return ((rx ** 2 - ry ** 2) * cos(theta) * sin(theta) -
            px * rx * sin(theta) + py * ry * cos(theta))


def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
    #The derivative of ellipe_tan_dot.
    return ((rx ** 2 - ry ** 2) * (cos(theta) ** 2 - sin(theta) ** 2) -
            px * rx * cos(theta) - py * ry * sin(theta))


def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    # Given a point (x, y), and an ellipse with major - minor axis (rx, ry), its center at (x0, y0), and with a counter clockwise rotation of `angle` degrees, will return the distance between the ellipse and the closest point on the ellipses boundary.
    x -= x0
    y -= y0
    if angle:
        # rotate the points onto an ellipse whose rx, and ry lay on the x, y
        # axis
        angle = -pi / 180. * angle
        x, y = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

    theta = atan2(rx * y, ry * x)
    while fabs(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
        theta -= ellipe_tan_dot(
            rx, ry, x, y, theta) / \
            ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    px, py = rx * cos(theta), ry * sin(theta)
    return ((x - px) ** 2 + (y - py) ** 2) ** .5

    
if __name__ == '__main__':
    # Example of image Boolean classifier (b=0.5, F_r = 50%, and T_p = 50h) 
    print(ImageOpenCV("snapshots/Output_B05_F050_T050.jpg"))

    # Analysis of 20 replicates for b=0.5, F_r = 50%, and T_p = 50h (Folder \replicas)
    ReplicaAnalysis()

    # Classifying all simulations from folder \snapshots
    Classify_all_simulations()
