import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme
    property var backend

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        radius: theme.cornerRadius
        border.color: theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme.padding
        spacing: theme.spacing

        Label {
            text: "Normal Preview"
            color: theme.textPrimary
            font.pixelSize: 18
            Layout.alignment: Qt.AlignLeft
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            color: theme.textSecondary
            text: !backend || backend.inspectSummary === ""
                  ? "Inspect a mesh to populate materials, UV sets, and UDIMs."
                  : backend.inspectSummary
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            visible: backend && backend.warningSummary !== ""
            color: theme.accent
            text: backend ? backend.warningSummary : ""
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: theme.surfaceAlt
            radius: theme.cornerRadius
            border.color: theme.border

            Image {
                id: previewImage
                anchors.fill: parent
                anchors.margins: theme.padding
                fillMode: Image.PreserveAspectFit
                visible: backend && backend.normalTextureUrl !== ""
                source: backend ? backend.normalTextureUrl : ""
            }

            Label {
                anchors.centerIn: parent
                visible: !previewImage.visible
                text: backend && backend.normalPath !== ""
                      ? "Normal map preview unavailable"
                      : "Select a normal map to preview"
                color: theme.textMuted
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
            }
        }

        CheckBox {
            text: "Apply normal map to viewport"
            checked: backend ? backend.normalEnabled : true
            enabled: backend && backend.normalTextureUrl !== ""
            onToggled: if (backend) backend.setNormalEnabled(checked)
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            color: theme.textSecondary
            visible: backend && backend.normalPreviewPath !== ""
            text: backend ? backend.normalPreviewPath : ""
        }

        Expander {
            visible: backend && backend.udimTileCount > 1
            Layout.fillWidth: true
            Layout.preferredHeight: implicitHeight
            text: "Tiles (advanced)"

            contentItem: ColumnLayout {
                anchors.fill: parent
                anchors.margins: theme.padding
                spacing: theme.spacing / 2

                Label {
                    text: backend && backend.selectedTile !== 0
                          ? "Active tile: " + backend.selectedTile
                          : "Active tile: —"
                    color: theme.textPrimary
                }

                Flow {
                    width: parent.width
                    spacing: theme.spacing / 2

                    Repeater {
                        model: backend ? backend.udimTiles : []

                        delegate: Button {
                            text: modelData
                            checkable: true
                            checked: backend && backend.selectedTile === Number(modelData)
                            onClicked: {
                                if (backend) backend.selectTile(Number(modelData))
                            }
                        }
                    }
                }
            }
        }
    }
}
