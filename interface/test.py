import kreps

c = kreps.InterfaceObject("test_parameter.json")
print("InterfaceObject version is: %s" % c.getVersion())
