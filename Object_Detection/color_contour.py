import time
import cv2
from colorsLib import getColor  # Library to map hex value of color to color name


def drawAllContour(img):

    # Subtracting 255 to make the white background black so that the results will improve
    img -= 255
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)


    # This returns error about Traceback (most recent call last): File "colourdetection.py", line 38, in image,
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) ValueError: need more
    # than 2 values to unpack
    # | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

if __name__ == "__main__":

    # Dictionary to store colors and their respective votes
    myClrDict = {}

    # Reading image to test
    MainImg = cv2.imread("Background Removed Images/BGR_IMG1.png")

    # Converting to RGB because opencv reads image to BGR format
    img = cv2.cvtColor(MainImg, cv2.COLOR_BGR2RGB)

    # Resizing the image to increase computation speed
    img = cv2.resize(img, (40, 40))

    height, width, _ = img.shape

    # Flattening the image matrix to a single array because handling of arrays is easy
    img_ary = img.reshape((height * width, 3))

    count = 0

    # Variable to store the background color
    bgClr = ""

    # Loop to check if the first 10 pixels are same (Means that the background is same)
    for j in range(10):
        if getColor(img_ary[j]) == getColor(img_ary[0]):
            count += 1

    # If they are same which means that background is constant. And save the background color which will be neglected
    # while finding the max color.
    if count == 10:
        bgClr = getColor(img_ary[0])

    # If pixels are same we say that background is not same which is violating our assumption
    else:
        print("background not same")
        exit()

    print("-------------------------------------Processing...Please Wait!!!------------------------------------")

    # Saving the current time
    start = time.time()

    # Starting a loop for each pixel in image. size is divided by 3 because previously we have flattened the matrix to
    # a single array so each index will have 3 values (R ,G ,B) representing a pixel
    for i in range(int(img_ary.size / 3)):

        # img_arr[i] will have the RGB value of pixel, we find the corresponding color from the colors library by
        # calling the getColor function lets say if RGB is (255,255,255) getColor will return "white"
        # using that color name as a key to store in dictionary
        myKey = (getColor(img_ary[i]))

        # if the color is background color, ignore it.
        if myKey == bgClr:
            continue
        try:  # If the key is already present in the dictionary the try block will run and increment the vote count
            myClrDict[myKey] += 1
        except: # Otherwise except will run and it will insert that key to the dictionary and put 1 vote there
            myClrDict[myKey] = 1

    # Printing the total time it took to process the image.
    print("---------------------------------Time Duration: ", time.time() - start, "--------------------------------"),

    # Sorting the dictionary to find the color with max votes.
    sorted_by_value = sorted(myClrDict.items(), key=lambda kv: kv[1], reverse=True)

    # Max voted color will be on top of the dictionary.
    # In sorted_by_value[x][y], x represents the key(color name), y represents the votes
    print("\nColor: ", sorted_by_value[0][0], "|  Votes: ", sorted_by_value[0][1])

    # 2nd Max will be on the [1][0]
    print("1st Nearest Color: ", sorted_by_value[1][0], "|  Votes: ", sorted_by_value[1][1])

    # 3rd Max will be on the [2][0]
    print("2nd Nearest Color: ", sorted_by_value[2][0], "|  Votes: ", sorted_by_value[2][1])

    # Drawing contour on MainImg
    drawAllContour(MainImg)

    # Displaying the image with Contours
    cv2.imshow("img", MainImg)
    # Save image with contours
    cv2.imwrite("Contour Mapped Images/CM_IMG1.png", MainImg)
    print("")

    print("-----------------Press Any Key To End------------------")
    cv2.waitKey()
    cv2.destroyAllWindows()

