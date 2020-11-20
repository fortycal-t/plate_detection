import sensor, time, image, tf


def sort_preds(pred):
    return pred[1]


sensor.reset()




img = image.Image("/temp/licenseplate.bmp", copy_to_fb = True)



plates = image.HaarCascade("plate/cascade.cascade", stages=10)

zero_cascade = image.HaarCascade("plate/0_cascade.cascade",stages=10)
five_cascade = image.HaarCascade("plate/5_cascade.cascade",stages=10)
six_cascade = image.HaarCascade("plate/6_cascade.cascade",stages=10)
s_cascade = image.HaarCascade("plate/s_cascade.cascade",stages=10)
g_cascade = image.HaarCascade("plate/g_cascade.cascade",stages=10)


net = "/plate/trained.tflite"
labels = [line.rstrip('\n') for line in open("/plate/labels.txt")]




found_plate = img.find_features(plates, threshold=1, scale_factor=1.5)

for f in found_plate:



    (x, y, w, h) = f

    f = (x, y, w, int(h*1.2))


    img.draw_rectangle(f)



    # find all the s's in the plate
    found_letter_s = img.find_features(s_cascade, threshold=1, scale_factor=1.5, roi=f)
    for z in found_letter_s:
        (x, y, w, h) = z

        z = (x, y, w, int(h*1.2))


        img.draw_rectangle(z)

    # find all the 0's in the plate
    found_zeros = img.find_features(zero_cascade, threshold=1, scale_factor=1.5, roi=f)

    for z in found_zeros:
        (x, y, w, h) = z

        z = (x, y, w, int(h*1.2))


        img.draw_rectangle(z)


    # find all the 5's in the plate
    found_fives = img.find_features(five_cascade, threshold=1, scale_factor=1.5, roi=f)

    for z in found_fives:
        (x, y, w, h) = z

        z = (x, y, w, int(h*1.2))


        img.draw_rectangle(z)

    # find all the 6's in the plate
    found_sixes = img.find_features(six_cascade, threshold=1, scale_factor=1.5, roi=f)

    for z in found_sixes:
        (x, y, w, h) = z

        z = (x, y, w, int(h*1.2))


        img.draw_rectangle(z)




    # find all the g's in the plate
    found_letter_g = img.find_features(g_cascade, threshold=1, scale_factor=1.5, roi=f)
    for z in found_letter_g:
        (x, y, w, h) = z

        z = (x, y, w, int(h*1.2))


        img.draw_rectangle(z)


    for obj in tf.classify(net, img, roi=f, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())

        predictions_list = list(zip(labels, obj.output()))

        sorted_list = list(zip(labels, obj.output()))
        sorted_list.sort(reverse=True, key=sort_preds)


        (x, y, w, h) = f
        img.draw_rectangle(x,y-10,w,10,(0,0,0),1,True)
        img.draw_string(x,y-10,sorted_list[0][0])

        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

        print ("detected plate is "+sorted_list[0][0] +sorted_list[1][0]+ sorted_list[2][0] +sorted_list[3][0] +sorted_list[4][0] + sorted_list[5][0])





# Flush FB
sensor.flush()

# Add a small delay to allow the IDE to read the flushed image.
time.sleep(100)
