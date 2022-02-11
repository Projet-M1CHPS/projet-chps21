import sys

# == neural-network interface ==
import kreps


# == ======================== ==


def onPrecisionChanged():
    print("onPrecisionChanged")


if len(sys.argv) < 2:
    print(f"Usage : python3 {sys.argv[0]} configuration_file_path\nNo configuration file given. \nStop.")
    exit(0)

print("InterfaceObject version is: %s" % kreps.getVersion())
c = kreps.NetworkInterface(sys.argv[1])
print("Made a networkInterface called !")
c.printJSONConfig()
# c.onPrecisionChanged(onPrecisionChanged)
c.createAndTrain()
