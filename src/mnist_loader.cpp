#include "../include/mnist_loader.h"

namespace load_data {

    Dataset read_input(const std::string& images_path, const std::string& labels_path) {
        Dataset data;
        std::cout << " |- Loading Images\n";
        data.images = read_images(images_path);
        std::cout << " |- Loading Labels\n";
        data.labels = read_labels(labels_path);
        std::cout << " |- File loaded\n\n";
        return data;
    }

    /*                 Image File structure
    #   [offset]    [type]      [value]     [desc]
    #   0000        int32       2051        Magic number for Images
    #   0004        int32       ?           Number of Images in file
    #   0008        int32       28          Number of Rows        
    #   00012       int32       28          Number of Columns
    #   00016       int8        ?           First pixel of first image
    #   00017       int8        ?           Second pixel
    #   ...         int8        ?           More pixels until EOF (784 bytes/pixels per image, flattened to row by row)
    */
    std::vector<Eigen::VectorXf> read_images(const std::string& path) {
        std::vector<Eigen::VectorXf> images;
        std::ifstream file(path, std::ios::binary);
        if(!file.is_open()) std::cerr << "Failed to open file\n";
        else {


            uint32_t magic_num = 0;
            uint32_t num_images = 0;
            uint32_t rows = 0;
            uint32_t cols = 0;

            file.read((char*)&magic_num, sizeof(magic_num));
            if(!file.good()) { std::cerr << "Error reading file\n"; return images; }
            magic_num = ntohl(magic_num);
            if(magic_num != 2051) { std::cerr << "Image file has incorrect magic number\n"; return images; }

            file.read((char*)&num_images, sizeof(num_images));
            if(!file.good()) { std::cerr << "Error reading file\n"; return images; }
            num_images = ntohl(num_images);
            images.reserve(num_images);

            file.read((char*)&rows, sizeof(rows));
            if(!file.good()) { std::cerr << "Error reading file\n"; return images; }
            rows = ntohl(rows);

            file.read((char*)&cols, sizeof(cols));
            if(!file.good()) { std::cerr << "Error reading file\n"; return images; }
            cols = ntohl(cols);

            for(int i = 0; i < num_images; i++) {
                Eigen::VectorXf image(rows*cols); //Should just be 28x28 for 784
                image.setZero();
                for(int j = 0; j < (rows * cols); j++) {
                    uint8_t pixel = 0;
                    file.read((char*)&pixel, sizeof(pixel));
                    image(j) = static_cast<float>(pixel) / 255.0f; //Go from [0, 255] to [0.0f, 1.0f] (black, white)
                }
                images.push_back(image);
            }
            std::cout << " |- " << num_images << " Images loaded\n";
        }

        return images;
    }

    /*                 Label File structure
    #   [offset]    [type]      [value]     [desc]
    #   0000        int32       2049        Magic number for labels
    #   0004        int32       ?           Number of Labels in file
    #   0008        int8        ?           First label, 0-9        
    #   0009        int8        ?           Second label...
    #   ...         int8        ?           More Labels until EOF
    */
    std::vector<uint8_t> read_labels(const std::string& path) {
        std::vector<uint8_t> labels;
        std::ifstream file(path, std::ios::binary);
        if(!file.is_open()) std::cerr << "Failed to open file\n";
        else {
            uint32_t magic_num = 0;
            uint32_t num_labels = 0;

            file.read((char*)&magic_num, sizeof(magic_num));
            if(!file.good()) { std::cerr << "Error reading file\n"; return labels; }
            magic_num = ntohl(magic_num);
            if(magic_num != 2049) { std::cerr << "Label file has incorrect magic number\n"; return labels; }

            file.read((char*)&num_labels, sizeof(num_labels));
            if(!file.good()) { std::cerr << "Error reading file\n"; return labels; }
            num_labels = ntohl(num_labels);

            labels.resize(num_labels);
            file.read((char*)labels.data(), num_labels);
            if(!file.good()) { std::cerr << "Error reading file\n"; return labels; }
            
            std::cout << " |- " << num_labels << " Labels loaded\n";
        }
        return labels;
    }
}