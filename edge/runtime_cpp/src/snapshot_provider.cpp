#include "visionops_runtime/snapshot_provider.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace visionops::runtime {

namespace {

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> output;
  int accumulator = 0;
  int bits = -8;
  for (const char ch : encoded) {
    if (ch == '=') break;
    const auto position = alphabet.find(ch);
    if (position == std::string_view::npos) continue;
    accumulator = (accumulator << 6) + static_cast<int>(position);
    bits += 6;
    if (bits >= 0) {
      output.push_back(static_cast<std::uint8_t>((accumulator >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return output;
}

constexpr std::array<int, 64> kZigZag{{
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63}};

constexpr std::array<int, 64> kLuminanceQuant{{
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99}};

constexpr std::array<int, 64> kChrominanceQuant{{
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99}};

constexpr std::array<std::uint8_t, 17> kDcLumBits{{0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}};
constexpr std::array<std::uint8_t, 12> kDcLumVals{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
constexpr std::array<std::uint8_t, 17> kDcChrBits{{0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}};
constexpr std::array<std::uint8_t, 12> kDcChrVals{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};

constexpr std::array<std::uint8_t, 17> kAcLumBits{{0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d}};
constexpr std::array<std::uint8_t, 162> kAcLumVals{{
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa}};

constexpr std::array<std::uint8_t, 17> kAcChrBits{{0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77}};
constexpr std::array<std::uint8_t, 162> kAcChrVals{{
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa}};

struct HuffmanEntry {
  std::uint16_t code{0};
  std::uint8_t size{0};
};

using HuffmanTable = std::array<HuffmanEntry, 256>;

template <std::size_t BitsN, std::size_t ValuesN>
HuffmanTable make_huffman_table(
    const std::array<std::uint8_t, BitsN>& bits,
    const std::array<std::uint8_t, ValuesN>& values) {
  HuffmanTable table{};
  std::uint16_t code = 0;
  std::size_t pos = 0;
  for (int length = 1; length <= 16; ++length) {
    const int count = bits[static_cast<std::size_t>(length)];
    for (int i = 0; i < count && pos < values.size(); ++i) {
      const std::uint8_t value = values[pos++];
      table[value] = HuffmanEntry{code, static_cast<std::uint8_t>(length)};
      ++code;
    }
    code <<= 1;
  }
  return table;
}

void write_u16(std::vector<std::uint8_t>& out, int value) {
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
  out.push_back(static_cast<std::uint8_t>(value & 0xFF));
}

void write_marker(std::vector<std::uint8_t>& out, std::uint8_t marker) {
  out.push_back(0xFF);
  out.push_back(marker);
}

class BitWriter {
 public:
  explicit BitWriter(std::vector<std::uint8_t>& out) : out_(out) {}

  void write_bits(std::uint16_t bits, std::uint8_t count) {
    for (int i = count - 1; i >= 0; --i) {
      current_ = static_cast<std::uint8_t>((current_ << 1) | ((bits >> i) & 1));
      ++used_;
      if (used_ == 8) flush_byte();
    }
  }

  void write_huffman(const HuffmanTable& table, std::uint8_t symbol) {
    const auto entry = table[symbol];
    write_bits(entry.code, entry.size);
  }

  void finish() {
    if (used_ > 0) {
      current_ = static_cast<std::uint8_t>((current_ << (8 - used_)) | ((1 << (8 - used_)) - 1));
      flush_byte();
    }
  }

 private:
  void flush_byte() {
    out_.push_back(current_);
    if (current_ == 0xFF) out_.push_back(0x00);
    current_ = 0;
    used_ = 0;
  }

  std::vector<std::uint8_t>& out_;
  std::uint8_t current_{0};
  std::uint8_t used_{0};
};

int category(int value) {
  int abs_value = value < 0 ? -value : value;
  int size = 0;
  while (abs_value > 0) {
    ++size;
    abs_value >>= 1;
  }
  return size;
}

std::uint16_t value_bits(int value, int size) {
  if (size == 0) return 0;
  if (value >= 0) return static_cast<std::uint16_t>(value);
  return static_cast<std::uint16_t>(value + ((1 << size) - 1));
}

std::array<int, 64> dct_quantize(
    const std::array<double, 64>& samples,
    const std::array<int, 64>& quant) {
  static bool initialized = false;
  static double cos_table[8][8]{};
  if (!initialized) {
    constexpr double pi = 3.14159265358979323846;
    for (int u = 0; u < 8; ++u) {
      for (int x = 0; x < 8; ++x) {
        cos_table[u][x] = std::cos(((2.0 * x + 1.0) * u * pi) / 16.0);
      }
    }
    initialized = true;
  }

  std::array<int, 64> out{};
  for (int v = 0; v < 8; ++v) {
    for (int u = 0; u < 8; ++u) {
      double sum = 0.0;
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          sum += samples[static_cast<std::size_t>(y * 8 + x)] * cos_table[u][x] * cos_table[v][y];
        }
      }
      const double cu = (u == 0) ? 0.7071067811865476 : 1.0;
      const double cv = (v == 0) ? 0.7071067811865476 : 1.0;
      const int index = v * 8 + u;
      out[static_cast<std::size_t>(index)] = static_cast<int>(std::round(0.25 * cu * cv * sum / quant[static_cast<std::size_t>(index)]));
    }
  }
  return out;
}

void encode_block(
    BitWriter& writer,
    const std::array<int, 64>& coeff,
    int& previous_dc,
    const HuffmanTable& dc_table,
    const HuffmanTable& ac_table) {
  const int dc = coeff[0];
  const int diff = dc - previous_dc;
  previous_dc = dc;
  const int dc_size = category(diff);
  writer.write_huffman(dc_table, static_cast<std::uint8_t>(dc_size));
  if (dc_size > 0) writer.write_bits(value_bits(diff, dc_size), static_cast<std::uint8_t>(dc_size));

  int zero_run = 0;
  for (int i = 1; i < 64; ++i) {
    const int value = coeff[static_cast<std::size_t>(kZigZag[static_cast<std::size_t>(i)])];
    if (value == 0) {
      ++zero_run;
      continue;
    }
    while (zero_run > 15) {
      writer.write_huffman(ac_table, 0xF0);
      zero_run -= 16;
    }
    const int ac_size = category(value);
    const std::uint8_t symbol = static_cast<std::uint8_t>((zero_run << 4) | ac_size);
    writer.write_huffman(ac_table, symbol);
    writer.write_bits(value_bits(value, ac_size), static_cast<std::uint8_t>(ac_size));
    zero_run = 0;
  }
  if (zero_run > 0) writer.write_huffman(ac_table, 0x00);
}

std::uint8_t channel_value(const ImageBuffer& image, int x, int y, int channel) {
  x = std::max(0, std::min(image.width - 1, x));
  y = std::max(0, std::min(image.height - 1, y));
  const std::size_t index = (static_cast<std::size_t>(y) * image.width + x) * 3 + channel;
  return index < image.data.size() ? image.data[index] : 0;
}

std::vector<std::uint8_t> encode_rgb_to_jpeg(const ImageBuffer& image) {
  if (!image_buffer_valid_rgb(image)) return {};
  if (image.pixel_format != "RGB888" && image.pixel_format != "BGR888" &&
      image.pixel_format != "RGB" && image.pixel_format != "BGR") {
    return {};
  }
  const bool bgr = image.pixel_format == "BGR888" || image.pixel_format == "BGR";
  const int width = image.width;
  const int height = image.height;
  if (width <= 0 || height <= 0 || width > 8192 || height > 8192) return {};

  std::vector<std::uint8_t> out;
  out.reserve(static_cast<std::size_t>(width) * height / 2 + 1024);

  write_marker(out, 0xD8);  // SOI

  write_marker(out, 0xE0);  // APP0 JFIF
  write_u16(out, 16);
  out.insert(out.end(), {'J', 'F', 'I', 'F', 0});
  out.push_back(1);
  out.push_back(1);
  out.push_back(0);
  write_u16(out, 1);
  write_u16(out, 1);
  out.push_back(0);
  out.push_back(0);

  write_marker(out, 0xDB);  // DQT
  write_u16(out, 2 + 65 * 2);
  out.push_back(0x00);
  for (int i = 0; i < 64; ++i) out.push_back(static_cast<std::uint8_t>(kLuminanceQuant[kZigZag[i]]));
  out.push_back(0x01);
  for (int i = 0; i < 64; ++i) out.push_back(static_cast<std::uint8_t>(kChrominanceQuant[kZigZag[i]]));

  write_marker(out, 0xC0);  // SOF0
  write_u16(out, 17);
  out.push_back(8);
  write_u16(out, height);
  write_u16(out, width);
  out.push_back(3);
  out.push_back(1); out.push_back(0x11); out.push_back(0);
  out.push_back(2); out.push_back(0x11); out.push_back(1);
  out.push_back(3); out.push_back(0x11); out.push_back(1);

  const auto write_dht = [&out](std::uint8_t table_class_and_id, const auto& bits, const auto& values) {
    write_marker(out, 0xC4);
    write_u16(out, static_cast<int>(2 + 1 + 16 + values.size()));
    out.push_back(table_class_and_id);
    for (int i = 1; i <= 16; ++i) out.push_back(bits[static_cast<std::size_t>(i)]);
    out.insert(out.end(), values.begin(), values.end());
  };
  write_dht(0x00, kDcLumBits, kDcLumVals);
  write_dht(0x10, kAcLumBits, kAcLumVals);
  write_dht(0x01, kDcChrBits, kDcChrVals);
  write_dht(0x11, kAcChrBits, kAcChrVals);

  write_marker(out, 0xDA);  // SOS
  write_u16(out, 12);
  out.push_back(3);
  out.push_back(1); out.push_back(0x00);
  out.push_back(2); out.push_back(0x11);
  out.push_back(3); out.push_back(0x11);
  out.push_back(0);
  out.push_back(63);
  out.push_back(0);

  const auto dc_lum = make_huffman_table(kDcLumBits, kDcLumVals);
  const auto ac_lum = make_huffman_table(kAcLumBits, kAcLumVals);
  const auto dc_chr = make_huffman_table(kDcChrBits, kDcChrVals);
  const auto ac_chr = make_huffman_table(kAcChrBits, kAcChrVals);

  BitWriter writer(out);
  int prev_y = 0;
  int prev_cb = 0;
  int prev_cr = 0;
  std::array<double, 64> y_block{};
  std::array<double, 64> cb_block{};
  std::array<double, 64> cr_block{};

  for (int by = 0; by < height; by += 8) {
    for (int bx = 0; bx < width; bx += 8) {
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          const int px = bx + x;
          const int py = by + y;
          const double c0 = channel_value(image, px, py, 0);
          const double c1 = channel_value(image, px, py, 1);
          const double c2 = channel_value(image, px, py, 2);
          const double r = bgr ? c2 : c0;
          const double g = c1;
          const double b = bgr ? c0 : c2;
          const std::size_t index = static_cast<std::size_t>(y * 8 + x);
          y_block[index] = 0.299 * r + 0.587 * g + 0.114 * b - 128.0;
          cb_block[index] = -0.168736 * r - 0.331264 * g + 0.5 * b;
          cr_block[index] = 0.5 * r - 0.418688 * g - 0.081312 * b;
        }
      }
      encode_block(writer, dct_quantize(y_block, kLuminanceQuant), prev_y, dc_lum, ac_lum);
      encode_block(writer, dct_quantize(cb_block, kChrominanceQuant), prev_cb, dc_chr, ac_chr);
      encode_block(writer, dct_quantize(cr_block, kChrominanceQuant), prev_cr, dc_chr, ac_chr);
    }
  }
  writer.finish();
  write_marker(out, 0xD9);  // EOI
  return out;
}

}  // namespace

const std::vector<std::uint8_t>& SnapshotProvider::fallback_snapshot_jpeg() const {
  // 1x1 像素 JPEG 内嵌于程序，不读取或提交图片文件。
  static const std::vector<std::uint8_t> image = decode_base64(
      "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
      "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
      "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
      "9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QA"
      "FBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QA"
      "FBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=");
  return image;
}

std::vector<std::uint8_t> SnapshotProvider::snapshot_jpeg(const ImageBuffer* image) const {
  if (image != nullptr && image_buffer_valid_rgb(*image)) {
    auto encoded = encode_rgb_to_jpeg(*image);
    if (!encoded.empty()) return encoded;
  }
  return fallback_snapshot_jpeg();
}

}  // namespace visionops::runtime
