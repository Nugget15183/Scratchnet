#include "mnist.h"

#include <fstream>
#include <cstdint>

namespace {
    //idx headers are big endian

    bool readBE32(ifstream& f, uint32_t& out) {
        //store read data
        unsigned char buf[4];

        //read the data into the buffer
        //if fail return false
        if (!f.read((char*)&buf, 4)) return false;

        //return the uint32 into the out obj
        out = (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) | (uint32_t(buf[2]) << 8) | buf[3];

        //return success
        return true;
    }
}

std::vector<Sample> mnist::load(const std::string &imagesPath, const std::string &labelsPath, size_t limit) {
    std::ifstream fi(imagesPath, std::ios::binary);
    std::ifstream fl(labelsPath, std::ios::binary);

    //if fail to read then return
    if (!fi || !fl) {return {};}

    uint32_t magicI;
    uint32_t countI;

    uint32_t rows;
    uint32_t cols;

    uint32_t magicL;
    uint32_t countL;

    //read all the variables needed later from the dataset
    if (!readBE32(fi, magicI) || !readBE32(fi, countI) || !readBE32(fi, rows) || !readBE32(fi, cols)) {
        return {};
    }

    if (!readBE32(fl, magicL) || !readBE32(fl, countL)) {
        return {};
    }

    if (magicI != 2051 || magicL != 2049 || countI != countL || rows != 28 || cols != 28) {
        return {};
    }

    //apply the limit
    size_t n = countI;

    if (limit > 0 && n > limit) {
        n = limit;
    }

    size_t px = rows * cols;
    std::vector<Sample> out(n);
    std::vector<unsigned char> buf(px);

    for (size_t i = 0; i < n; i++) {
        int label = 0;

        //read the data and make sure its vaild
        if (!fi.read((char*)buf.data(), px) || !fl.read((char*)&label, 1) || label > 9) {
            return {};
        }

        out[i].x.resize(px);

        for (size_t k = 0; k < px; k++) {
            out[i].x[k] = buf[k] / 255.0f;
        }

        //one hot
        out[i].y.assign(10, 0.0f);
        out[i].y[label] = 1.0f;
    }
    return out;
}