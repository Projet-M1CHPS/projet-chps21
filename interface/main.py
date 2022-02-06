import kreps

c = kreps.NetworkInterface("parameters.json")
print("Made a networkInterface called !")
c.printJSONConfig();
print("InterfaceObject version is: %s" % kreps.getVersion())
