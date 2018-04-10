# USAGE
# python pi_surveillance.py --conf conf.json

# import the necessary packages
from pyimagesearch.tempimage import TempImage
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dropbox import Dropbox
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import logging
import sys

def get_dropbox_client(access_token=""):
    """
    Given an access_token, return a Dropbox client.  If the access_token is an
    empty string, prompt user to authenticate the app to Dropbox and create a
    client using the returned access_token.
    
    :param access_token: string
    :return: Dropbox
    """

    if access_token == "":
        # connect to dropbox and start the session authorization process
        flow = DropboxOAuth2FlowNoRedirect(conf["dropbox_key"], conf["dropbox_secret"])
        print("[INFO] Authorize this application: {}".format(flow.start()))
        authCode = input("Enter auth code here: ").strip()

        # finish the authorization and grab the Dropbox client
        access_token = flow.finish(authCode).access_token

        # Display the access token so it can be stored for later use.
        # @todo: have the program write this out somewhere so it doesn't have to be done manually.
        logging.info("################")
        logging.info("Your access token is {}. set this value in conf.json so you don't have to reauthenticate \
    			each time the program is run".format(access_token))
        logging.info("################")

    client = Dropbox(access_token)
    logging.info("Dropbox client connected.")
    return client

if __name__ == "__main__":
    try:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--conf", required=True,
                        help="path to the JSON configuration file")
        args = vars(ap.parse_args())
        conf = json.load(open(args["conf"]))

        # filter warnings and initialize the Dropbox client
        warnings.filterwarnings("ignore")

        # check to see if the Dropbox should be used
        client = None
        if conf["use_dropbox"]:
            logging.info("[INFO] Using Dropbox")
            client = get_dropbox_client(conf["dropbox_access_token"])

        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.resolution = tuple(conf["resolution"])
        camera.framerate = conf["fps"]
        rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

        # allow the camera to warmup, then initialize the average frame, last
        # uploaded timestamp, and frame motion counter
        logging.info("[INFO] warming up...")
        time.sleep(conf["camera_warmup_time"])
        avg = None
        lastUploaded = datetime.datetime.now()
        motionCounter = 0

        # capture frames from the camera
        for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image and initialize
            # the timestamp and occupied/unoccupied text
            frame = f.array
            timestamp = datetime.datetime.now()
            text = "Unoccupied"

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the average frame is None, initialize it
            if avg is None:
                logging.info("[INFO] starting background model...")
                avg = gray.copy().astype("float")
                rawCapture.truncate(0)
                continue

            # accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

            # threshold the delta image, dilate the thresholded image to fill
            # in holes, then find contours on thresholded image
            thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                                   cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # As of the 3.2 version of OpenCV, findCountours returns three
            # values but we only care about the middle one.
            # See http://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/#comment-364523
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < conf["min_area"]:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

            # draw the text and timestamp on the frame
            ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)

            # check to see if the room is occupied
            if text == "Occupied":
                # check to see if enough time has passed between uploads
                if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
                    # increment the motion counter
                    motionCounter += 1

                    # check to see if the number of frames with consistent motion is
                    # high enough
                    if motionCounter >= conf["min_motion_frames"]:
                        # check to see if dropbox sohuld be used
                        if conf["use_dropbox"]:
                            # write the image to temporary file
                            t = TempImage()
                            cv2.imwrite(t.path, frame)

                            # upload the image to Dropbox and cleanup the tempory image
                            logging.info("[UPLOAD] {}".format(ts))
                            path = "/{base_path}/{timestamp}.jpg".format(
                                base_path=conf["dropbox_base_path"], timestamp=ts)
                            client.files_upload(open(t.path, "rb").read(), path)
                            t.cleanup()

                        # update the last uploaded timestamp and reset the motion
                        # counter
                        lastUploaded = timestamp
                        motionCounter = 0

            # otherwise, the room is not occupied
            else:
                motionCounter = 0

            # check to see if the frames should be displayed to screen
            if conf["show_video"]:
                # display the security feed
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
    except (KeyboardInterrupt):
        logging.info("Received keyboard interrupt. Shutting down.")
        exit()
