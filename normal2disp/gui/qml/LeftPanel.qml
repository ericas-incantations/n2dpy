import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3

Item {
    id: root
    property var theme
    property var backend

    function cleanPath(url) {
        if (!url) {
            return ""
        }
        var stringValue = url.toString()
        if (stringValue.startsWith("file:///")) {
            stringValue = stringValue.substring(8)
        } else if (stringValue.startsWith("file://")) {
            stringValue = stringValue.substring(7)
        }
        return decodeURIComponent(stringValue)
    }

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        radius: theme.cornerRadius
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: theme.padding
        clip: true

        ColumnLayout {
            id: content
            spacing: theme.spacing
            width: parent.width

            Label {
                text: "Project"
                color: theme.textSecondary
                font.pixelSize: 18
                Layout.alignment: Qt.AlignLeft
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: theme.surfaceAlt
                    radius: theme.cornerRadius
                    border.color: theme.border
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.padding
                    spacing: theme.spacing

                    Label {
                        text: "Mesh"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: {
                                if (backend) {
                                    backend.browseMesh(backend.meshPath)
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: !backend || backend.meshPath === "" ? "No mesh selected" : backend.meshPath
                        }
                    }

                    Label {
                        text: "Normal map"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: normalDialog.open()
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: !backend || backend.normalPath === ""
                                  ? "No normal map selected"
                                  : backend.normalPath
                        }
                    }

                    Label {
                        text: "Output directory"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: outputDialog.open()
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: !backend || backend.outputDirectory === ""
                                  ? "No output folder selected"
                                  : backend.outputDirectory
                        }
                    }
                }
            }

            Label {
                text: "Bake options"
                color: theme.textSecondary
                font.pixelSize: 18
                Layout.alignment: Qt.AlignLeft
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: theme.surfaceAlt
                    radius: theme.cornerRadius
                    border.color: theme.border
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.padding
                    spacing: theme.spacing

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Material"
                            color: theme.textPrimary
                        }

                        ComboBox {
                            id: materialCombo
                            Layout.fillWidth: true
                            model: backend ? backend.materials : []
                            textRole: "name"
                            valueRole: "id"
                            enabled: backend && backend.materialCount > 0
                        }
                    }

                    Label {
                        visible: backend && backend.materialCount === 0
                        text: "No materials detected"
                        color: theme.textMuted
                        font.pixelSize: 12
                        Layout.fillWidth: true
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "UV Set"
                            color: theme.textPrimary
                        }

                        ComboBox {
                            id: uvSetCombo
                            Layout.fillWidth: true
                            model: backend ? backend.uvSets : []
                            textRole: "name"
                            enabled: backend && backend.uvSetCount > 0
                        }
                    }

                    Label {
                        visible: backend && backend.uvSetCount === 0
                        text: "No UV sets detected"
                        color: theme.textMuted
                        font.pixelSize: 12
                        Layout.fillWidth: true
                    }

                    CheckBox {
                        id: yDownCheck
                        text: "Y is down"
                        checked: false
                        font.pixelSize: 14
                        Layout.alignment: Qt.AlignLeft
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Amplitude"
                            color: theme.textPrimary
                        }

                        Slider {
                            id: amplitudeSlider
                            from: 0
                            to: 10
                            value: 1
                            Layout.fillWidth: true
                            onMoved: amplitudeSpin.value = value
                        }

                        SpinBox {
                            id: amplitudeSpin
                            from: 0
                            to: 10
                            stepSize: 0.1
                            value: amplitudeSlider.value
                            editable: true
                            Layout.preferredWidth: 80
                            onValueModified: amplitudeSlider.value = value
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Max slope"
                            color: theme.textPrimary
                        }

                        Slider {
                            id: slopeSlider
                            from: 1
                            to: 60
                            value: 10
                            Layout.fillWidth: true
                            onMoved: slopeSpin.value = value
                        }

                        SpinBox {
                            id: slopeSpin
                            from: 1
                            to: 60
                            stepSize: 1
                            value: slopeSlider.value
                            editable: true
                            Layout.preferredWidth: 80
                            onValueModified: slopeSlider.value = value
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Normalization"
                            color: theme.textPrimary
                        }

                        ComboBox {
                            id: normalizationCombo
                            Layout.fillWidth: true
                            model: ["Auto", "XYZ", "XY", "None"]
                            currentIndex: 0
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "CG Tol"
                            color: theme.textPrimary
                        }

                        TextField {
                            id: cgTolField
                            text: "1e-6"
                            Layout.fillWidth: true
                            placeholderText: "1e-6"
                            inputMethodHints: Qt.ImhNoPredictiveText | Qt.ImhPreferNumbers
                        }

                        Label {
                            text: "Max iters"
                            color: theme.textPrimary
                        }

                        SpinBox {
                            id: cgMaxSpin
                            from: 1
                            to: 200000
                            value: 10000
                            Layout.preferredWidth: 120
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Processes"
                            color: theme.textPrimary
                        }

                        SpinBox {
                            id: processSpin
                            from: 0
                            to: 32
                            value: 0
                            Layout.preferredWidth: 80
                        }
                    }

                    CheckBox {
                        id: deterministicCheck
                        text: "Deterministic"
                        checked: false
                        Layout.alignment: Qt.AlignLeft
                    }

                    CheckBox {
                        id: sidecarCheck
                        text: "Export sidecars"
                        checked: false
                        Layout.alignment: Qt.AlignLeft
                    }

                    Button {
                        text: backend && backend.inspectRunning ? "Inspecting…" : "Inspect Mesh"
                        Layout.fillWidth: true
                        enabled: backend && backend.meshPath !== "" && !backend.inspectRunning
                        onClicked: {
                            if (backend) {
                                backend.runInspect(backend.meshPath)
                            }
                        }
                    }

                    Button {
                        text: backend && backend.bakeRunning ? "Baking…" : "Bake"
                        Layout.fillWidth: true
                        highlighted: true
                        enabled: backend
                                  && backend.meshPath !== ""
                                  && backend.normalPath !== ""
                                  && backend.outputDirectory !== ""
                                  && !backend.bakeRunning
                        onClicked: {
                            if (!backend) {
                                return
                            }
                            backend.runBake({
                                "uvSet": uvSetCombo.currentIndex >= 0 ? uvSetCombo.currentText : "",
                                "yIsDown": yDownCheck.checked,
                                "normalization": normalizationCombo.currentText,
                                "amplitude": amplitudeSlider.value,
                                "maxSlope": slopeSlider.value,
                                "cgTol": cgTolField.text,
                                "cgMaxIter": cgMaxSpin.value,
                                "deterministic": deterministicCheck.checked,
                                "processes": processSpin.value,
                                "exportSidecars": sidecarCheck.checked
                            })
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: normalDialog
        title: "Select normal map"
        nameFilters: ["Images (*.png *.exr *.tif *.tiff)", "All files (*)"]
        onAccepted: {
            if (backend) {
                backend.setNormalPath(cleanPath(selectedFile))
            }
        }
    }

    FolderDialog {
        id: outputDialog
        title: "Select output directory"
        onAccepted: {
            if (backend) {
                backend.setOutputDirectory(cleanPath(selectedFolder))
            }
        }
    }
}
