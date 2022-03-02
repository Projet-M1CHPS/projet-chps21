#include "clPlatformSelector.hpp"
#include <signal.h>
#include <unistd.h>

namespace utils {

  std::unique_ptr<clWrapper> clPlatformSelector::execute() {
    clPlatformSelector selector;
    selector.loop();
    return std::make_unique<clWrapper>(selector.platforms[selector.selected_platform],
                                       selector.selected_default_device);
  }

  clPlatformSelector::clPlatformSelector()
      : selected_platform(0), selected_default_device(0), current_platform(0), current_device(0),
        selected_window(0) {
    cl::Platform::get(&platforms);

    for (auto &platform : platforms) {
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_map[&platform]);
    }
    initscr();

    start_color();
    init_pair(1, COLOR_GREEN, COLOR_BLACK);
    init_pair(2, COLOR_BLACK, COLOR_CYAN);
    init_pair(3, COLOR_BLACK, COLOR_GREEN);
    init_pair(4, COLOR_BLACK, COLOR_WHITE);

    keypad(stdscr, TRUE);
    noecho();
    curs_set(0);

    selected_color = COLOR_PAIR(1);
    current_color = COLOR_PAIR(2);
    selected_current_color = COLOR_PAIR(3);
    current_background_color = COLOR_PAIR(4);
    size_t col_size = COLS / 3;

    platform_win = newwin(LINES - 1, col_size, 0, 0);
    devices_list_win = newwin(LINES - 1, col_size, 0, col_size);
    device_details_win = newwin(LINES - 1, col_size, 0, COLS - col_size);
  }

  void clPlatformSelector::cleanup() {
    refresh();
    delwin(platform_win);
    delwin(devices_list_win);
    delwin(device_details_win);
    endwin();
  }

  clPlatformSelector::~clPlatformSelector() { cleanup(); }

  bool clPlatformSelector::handleInput() {
    int ch = getch();
    if (ch == KEY_RIGHT) {
      selected_window = std::min(1, selected_window + 1);
    } else if (ch == KEY_LEFT) {
      // Loops around to the right
      selected_window = std::max(0, selected_window - 1);
    } else if (ch == KEY_UP) {
      current_device = 0;
      switch (selected_window) {
        case 0:
          current_platform = std::max(current_platform - 1, 0);
          break;
        case 1:
          current_device = std::max(current_device - 1, 0);
          break;
      }
    } else if (ch == KEY_DOWN) {
      current_device = 0;
      switch (selected_window) {
        case 0:
          current_platform = std::min(platforms.size() - 1, (size_t) current_platform + 1);
          break;
        case 1:
          current_device = std::min(devices_map[&platforms[current_platform]].size() - 1,
                                    (size_t) current_device + 1);
          break;
      }
    } else if (ch == 10) {
      switch (selected_window) {
        case 0:
          selected_platform = current_platform;
          break;
        case 1:
          selected_default_device = current_device;
          break;
      }
    } else if (ch == 'x') {
      cleanup();
      return true;
    }
    return false;
  }

  void clPlatformSelector::loop() {
    bool done = false;
    while (not done) {
      display();
      done = handleInput();
    }
  }


  void clPlatformSelector::displayPlatforms() {
    wmove(platform_win, 0, 0);
    for (size_t i = 0; i < platforms.size(); i++) {
      std::string name = platforms[i].getInfo<CL_PLATFORM_NAME>();
      if (i == current_platform and selected_window == 0) {
        wattron(platform_win, current_color);
      } else if (i == current_platform) {
        wattron(platform_win, current_background_color);
      } else if (i == selected_platform) {
        wattron(platform_win, selected_color);
      }
      wprintw(platform_win, "%s\n", name.c_str());
      wattrset(platform_win, 0);
    }
  }

  void clPlatformSelector::displayDevices() {
    wmove(devices_list_win, 0, 0);
    auto &devices = devices_map[&platforms[current_platform]];
    for (size_t i = 0; i < devices.size(); i++) {
      std::string name = devices[i].getInfo<CL_DEVICE_NAME>();

      if (i == current_device and selected_window == 1) {
        wattron(devices_list_win, current_color);
      } else if (current_platform == selected_platform) {
        if (i == current_device and i == selected_default_device) {
          wattron(devices_list_win, selected_current_color);
        } else if (i == selected_default_device) {
          wattron(devices_list_win, selected_color);
        }
      } else if (i == current_device) {
        wattron(devices_list_win, current_background_color);
      }

      wprintw(devices_list_win, "%s\n", name.c_str());
      wattrset(devices_list_win, 0);
    }
  }

  void clPlatformSelector::displayDetails() {
    wmove(device_details_win, 0, 0);
    auto &devices = devices_map[&platforms[current_platform]];
    auto &device = devices[current_device];

    std::string name = device.getInfo<CL_DEVICE_NAME>();
    wprintw(device_details_win, "Name: %s\n", name.c_str());

    std::string type = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ? "CPU" : "GPU";
    wprintw(device_details_win, "Type: %s\n", type.c_str());

    std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
    wprintw(device_details_win, "Vendor: %s\n", vendor.c_str());

    std::string driver = device.getInfo<CL_DRIVER_VERSION>();
    wprintw(device_details_win, "Driver: %s\n", driver.c_str());

    std::string version = device.getInfo<CL_DEVICE_VERSION>();
    wprintw(device_details_win, "Version: %s\n", version.c_str());

    std::string profile = device.getInfo<CL_DEVICE_PROFILE>();
    wprintw(device_details_win, "Profile: %s\n", profile.c_str());

    size_t max_alloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    wprintw(device_details_win, "Max alloc: %zu\n", max_alloc);

    bool async =
            device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    wprintw(device_details_win, "Async: %s\n", async ? "yes" : "no");

    bool compiler_available = device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
    wprintw(device_details_win, "Compiler available: %s\n", compiler_available ? "yes" : "no");

    wprintw(device_details_win, "\n\n\n");
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    wprintw(device_details_win, "Extensions: %s\n", extensions.c_str());
  }

  void clPlatformSelector::display() {
    displayPlatforms();
    displayDevices();
    displayDetails();
    refresh();
    wrefresh(platform_win);
    wrefresh(devices_list_win);
    wrefresh(device_details_win);
  }
}   // namespace utils
