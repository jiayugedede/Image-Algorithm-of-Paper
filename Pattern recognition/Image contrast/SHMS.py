    def img_contrast_enhance_SHMS(self):
        
        """
        self.pic: must be a gray scale image
        Tip! : The max gray scale L may be not 255, it would be max gray level in current gray
               scale image.
        :return:
            Remapping a more bight gray scale image which contrast to input image. This function return four image
            which contain histogram equalizatino, piecewise histogram equalization and so on.
            
            reference:
                Chang Y C , Chang C M . A simple histogram modification scheme for contrast enhancement[J].
                IEEE Transactions on Consumer Electronics, 2010, 56(2):737-742.
        """
        def count_elements(img):
            """
            *** numpy的ravel函数功能是将多维数组降为一维数组
            :param img:
                2D array image --> gray scale image.
                :type
                gray scale  image array.
            :return:
                A dict of histogram of input gray scale image.
            """
            img = np.float32(img)
            one_dimension_img = img.ravel()
            # hist = {}
            # for i in one_dimension_img:
            #     hist[i] = hist.get(i, 0) + 1
            # hist = Counter(one_dimension_img) #counter original API of python
            histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
            return histogram, bin_edges

        def probability_density_functio(histogram, max_gray_scale, N):
            """
            :param histogram:
                Finding Kth gray level of histogram.
                :type
                dictionary
            :param max_gray_scale:
                :type
                int
            :param N:
                the total number of N pixels with gray levels in the range [ 0, L-1]
                :type
                int
            :return:
                probability density function(PDF) --> p(k)
                :type
                dictionary
            """
            pk = {}
            for i in range(max_gray_scale):
                pk[i] = histogram[i] / N
            return pk

        def cumulative_distribution_function(kth, pdf_list):
            """
            :param kth:
                kth gray level
                :type
                int
            :param pdf:
                probability density function(PDF) --> pk
                :type
                The list which is contain pk from 0th to kth of histogram.
            :return:
                cumulative_distribution_function(CDF) --> ck
                :type
                float
            """
            ck = 0
            for i in range(kth):
                ck += pdf_list[i]
            return ck

        def histogram_remapping(max_gray_level, cdf_list):
            """
            :param max_gray_level:
                max gray level of
                :type
                int
            :param cdf_list:
                cumulative_distribution_function(CDF) --> cdf_list
                :type:
                list
            :return:
                remapped gray scale to output image
                :type:
                dictionary
            """
            tk = {}
            for i in range(max_gray_level):
                tk_value = max_gray_level * cdf_list[i]
                if tk_value > 255:
                    tk_value = 255
                tk[i] = math.ceil(tk_value)
            return tk

        def mean_of_image(max_gray_level, pdf_list):
            """
            :param max_gray_level:
                max_gray_level would not be 255, it is may be the max gray scale of this image
                :type
                int
            :param pdf_list:
                pdf_list contain many pdf params in list
                :type
                list
            :return:
                mean of image
                :type
                int
            """
            m = 0
            for i in range(max_gray_level):
                m += i * pdf_list[i]
            return m

        def BBHE_L_remapping(mean_gray_scale, ck_list, magnify=1.0):
            """
            This is a half of image that gray level just below mean in original image.
            :param mean_gray_scale:
                mean gray scale in current image
                :type
                int
            :param ck_list:
                 cumulative distribution function list
                 :type
                 list
            :return:
                Remapping gray scale to output image, which contain original gray scale and remapped gray scale.
                :type
                dict
            """
            tk = {}  # key is gray scale, value is remapping value.
            for i in range(mean_gray_scale):
                tk[i] = ck_list[i] * mean_gray_scale * magnify
            return tk

        def BBHE_U_remapping(mean_gray_scale, max_gray_level, ck_list):
            """
            This is a half of image that gray level just below mean in original image.
            :param mean_gray_scale:
                mean gray scale in current image
                :type
                int

            :param max_gray_level:
                max_gray_level would not be 255, it is may be the max gray scale of this image
                :type
                int

            :param ck_list:
                 cumulative distribution function list
                 :type
                 list

            :return:
                Remapping gray scale to output image, which contain original gray scale and remapped gray scale.
                :type
                dict

            """
            tk = {}  # key is gray scale, value is remapping value.
            for i in range(max_gray_level - mean_gray_scale-1):
                tk[i+mean_gray_scale] = ck_list[i] * \
                    (max_gray_level - mean_gray_scale-2) + mean_gray_scale+1
            return tk

        def img_enhance_pic(img_copy, remapped_dic):
            # print("...................................")
            rows, cols = img_copy.shape
            pic_copy = deepcopy(img_copy)
            for i in range(rows):
                for j in range(cols):
                    pic_copy[i, j] = int(remapped_dic[pic_copy[i, j]])
            return pic_copy

        def histogram_equalization(pic_copy, max_l, cdf_list):
            remapped_t = histogram_remapping(max_l, cdf_list)
            enhance_img = img_enhance_pic(pic_copy, remapped_t)
            return enhance_img, remapped_t

        def piecewise_histogram_equalization(pic_copy, pdf, cdf_list, magnify=1.0):
            """


            Parameters
            ----------
            pic_copy : picture
                src pic.
            pdf : list
                probability density functio.
            cdf_list : list
                cumulative distribution function.

            Returns
            -------
            piecewise_enhance_img : TYPE
                DESCRIPTION.

            """
            mean_img_value = mean_of_image(
                max_l, pdf)  # decomposed histogram equlization

            mean_img_value_int = math.ceil(mean_img_value)
            # print("mean_img_value_int",mean_img_value_int)
            BBHE_L_index = np.arange(start=0, stop=mean_img_value_int+1)
            BBHE_L_index_list = BBHE_L_index.tolist()

            BBHE_U_index = np.arange(start=mean_img_value_int+1, stop=max_l)
            BBHE_U_index_list = BBHE_U_index.tolist()

            # print("BBHE_L_index_list", BBHE_L_index_list)
            # print("BBHE_U_index_list",BBHE_U_index_list)

            cdf_list_nparray = np.array(cdf_list)

            cdf_BBHE_L = cdf_list_nparray[BBHE_L_index_list]
            cdf_BBHE_U = cdf_list_nparray[BBHE_U_index_list]

            piecewise_L = BBHE_L_remapping(
                mean_img_value_int, cdf_BBHE_L, magnify)
            piecewise_U = BBHE_U_remapping(
                mean_img_value_int, max_l, cdf_BBHE_U)
            # print(cdf_BBHE_U)
            piecewise_regulation = {}
            piecewise_regulation.update(piecewise_L)
            piecewise_regulation.update(piecewise_U)

            piecewise_enhance_img = img_enhance_pic(
                pic_copy, piecewise_regulation)
            return piecewise_enhance_img, piecewise_regulation

        def SHMS_histogram_equalization(pic_copy, histogram, magnify=1):
            """


            Parameters
            ----------
            pic_copy : image
                a gray scale image.
            histogram : list
                a gray scale image contain the number of each gray level.

            Returns
            -------
            SHMS_enhance_img : image
                piecewise SHMS histogram enhance method.

            """
            delta_x = []
            for i in range(len(histogram)-1):
                delta_x.append(histogram[i+1] - histogram[i])
            delta_x = np.array(delta_x)
            delta_x_zero = delta_x != 0
            delta_x_zero_list = delta_x_zero.tolist()

            start_index = 0  # The rising edge signal is detected here
            for i in range(len(delta_x_zero_list)-1):
                if delta_x_zero_list[i+1] - delta_x_zero_list[i] > 0 and i < len(delta_x_zero_list):
                    if histogram[i] == 0:
                        start_index = i+1  # return delta != 0 parametes.

            if histogram[start_index] == 0:
                start_index += 1  # This is also a non-zero index

            end_index = 0  # Here the rising edge signal is detected in reverse
            for i in range(len(delta_x_zero)):
                if delta_x_zero[-i] == False and delta_x_zero[-(i+1)] == True and histogram[-i] == 0:
                    # return delta != 0 parametes.
                    end_index = len(delta_x_zero)-(i+1)

            if histogram[end_index] == 0:
                end_index -= 1  # This is also a non-zero index

            histogram[start_index+1] = 0
            histogram[end_index] = min(
                histogram[end_index], histogram[end_index-1])

            for i in range(max_l):
                hist_dict[bin_edges[i]] = histogram[i]
            pdf = probability_density_functio(histogram, 256, total_number)

            cdf = {}
            cdf_list = []
            for i in range(max_l):
                cdf_value = cumulative_distribution_function(i, pdf)
                cdf[i] = cdf_value
                cdf_list.append(cdf_value)

            SHMS_enhance_img, piecewise_regulation = piecewise_histogram_equalization(
                pic_copy, pdf, cdf_list, magnify=magnify)
            return SHMS_enhance_img

        def SHMS_Histogram_edited(histogram):
            """

            Parameters
            ----------

            histogram : list
                Only edit histogram, and use edited histogram do remapping .
            Returns
            -------
            edited histogram list/array.

            """

            delta_x = []
            for i in range(len(histogram)-1):
                delta_x.append(histogram[i+1] - histogram[i])
            delta_x = np.array(delta_x)
            delta_x_zero = delta_x != 0
            delta_x_zero_list = delta_x_zero.tolist()

            start_index = 0  # The rising edge signal is detected here
            start_flag = 0
            for i in range(len(delta_x_zero_list)-1):
                if delta_x_zero_list[i+1] - delta_x_zero_list[i] > 0 and i < len(delta_x_zero_list):
                    if histogram[i] == 0 and start_flag <= 0:
                        start_index = i+1  # return delta != 0 parametes.
                        start_flag += 1

            if histogram[start_index] == 0:
                start_index += 1  # 这个是非零值得索引

            end_index = 0  # Here the rising edge signal is detected in reverse
            end_flag = 0
            for i in range(len(delta_x_zero)):
                if delta_x_zero[-i] == False and delta_x_zero[-(i+1)] == True and histogram[-i] == 0:
                    if end_flag <= 0:
                        # return delta != 0 parametes.
                        end_index = len(delta_x_zero)-(i+1)
                        end_flag += 1

            if histogram[end_index] == 0:
                end_index -= 1  # This is also a non-zero index

            histogram[start_index+1] = 0
            histogram[end_index] = min(
                histogram[end_index], histogram[end_index-1])
            return histogram

        def Using_SHMS_do_histogram_equalization(pic_copy, histogram):
            histogram_edited = SHMS_Histogram_edited(histogram)

            hist_dict = {}
            for i in range(max_l):
                hist_dict[bin_edges[i]] = histogram_edited[i]
            pdf = probability_density_functio(
                histogram_edited, 256, total_number)

            cdf = {}
            cdf_list = []
            for i in range(max_l):
                cdf_value = cumulative_distribution_function(i, pdf)
                cdf[i] = cdf_value
                cdf_list.append(cdf_value)
            enhance_img, remapped_t = histogram_equalization(
                pic_copy, max_l, cdf_list)
            return enhance_img

        # *****************************************************************
        # 主体代码实现
        # *****************************************************************
        max_l = 256
        # bin_edges是gray level, histogram是每个灰度级像素的数量。
        histogram, bin_edges = count_elements(self.pic)
        bin_edges = bin_edges * max_l
        rows, cols = self.pic.shape

        pic_copy = deepcopy(self.pic)
        pic_copy = pic_copy*255
        pic_copy = np.uint8(pic_copy)

        total_number = rows * cols

        bin_edges = np.array(bin_edges).astype(np.int)
        hist_dict = {}
        for i in range(max_l):
            hist_dict[bin_edges[i]] = histogram[i]
        pdf = probability_density_functio(histogram, 256, total_number)

        cdf = {}
        cdf_list = []
        for i in range(max_l):
            cdf_value = cumulative_distribution_function(i, pdf)
            cdf[i] = cdf_value
            cdf_list.append(cdf_value)

        # enhance——img是直方图均衡化的结果图片
        enhance_img, HE_remapping = histogram_equalization(
            pic_copy, max_l, cdf_list)

        # 这里是分段式图像增强算法
        piecewise_enhance_img, piecewise_remapping = piecewise_histogram_equalization(
            pic_copy, pdf, cdf_list, magnify=3)

        # SHMS分段式图像增强对比度算法
        SHMS_enhance_img = SHMS_histogram_equalization(
            pic_copy, histogram, magnify=3)

        # 使用SHMS直方图进行原始的直方图均衡化
        SHMS_equzalization = Using_SHMS_do_histogram_equalization(
            pic_copy, histogram)

        return enhance_img, piecewise_enhance_img, SHMS_enhance_img, SHMS_equzalization
        #SHMS algorithm is proposed in the paper.
