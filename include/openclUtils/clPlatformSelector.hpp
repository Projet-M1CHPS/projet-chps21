#pragma once
#include "clWrapper.hpp"
#include <ncurses.h>

namespace utils {
  /**
   * @brief Provides a simple interface to select a platform and default device
   *
   */
  class clPlatformSelector {
  public:
    /**
     * @brief Display a cli interface to select a platform and default device. When the user exit,
     * return a cl wrapper containing the selected platform and device
     * @return
     */
    static std::unique_ptr<clWrapper> execute();

    ~clPlatformSelector();

    /**
     * @brief Display a cli interface to select a platform and default device. When the user exit
     * the cli, calls initOpenCL() on the selected platform
     */
    static void initOpenCL() {
      auto wrapper = execute();
      clWrapper::initOpenCL(*wrapper);
    }


  private:
    clPlatformSelector();

    void displayPlatforms();
    void displayDevices();
    void displayDetails();
    void display();

    void cleanup();

    bool handleInput();

    void loop();

    std::vector<cl::Platform> platforms;
    std::map<cl::Platform *, std::vector<cl::Device>> devices_map;

    int selected_platform;
    int selected_default_device;
    int current_platform;
    int current_device;


    WINDOW *platform_win;
    WINDOW *devices_list_win;
    WINDOW *device_details_win;
    int selected_color, current_color, selected_current_color;
    int current_background_color;
    int selected_window;
  };

}   // namespace utils